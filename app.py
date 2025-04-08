import time
import math 
import os
import random
import pandas as pd
import logging
from collections import defaultdict
from flask import Flask, render_template, request, jsonify
import json
import yfinance as yf
from functools import lru_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("stock_app.log", mode='a'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = os.path.join(os.path.expanduser("~"), "stock_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "sp500_cache.json")
CACHE_DURATION = 86400  # 1 day in seconds
os.makedirs(CACHE_DIR, exist_ok=True)

class DataCache:
    """Centralized caching mechanism for stock data."""
    def __init__(self):
        self.memory_cache = {}
        self.ticker_cache = {}
    
    def get(self, key, duration=3600):
        """Get data from memory cache if valid."""
        if key in self.memory_cache:
            if time.time() - self.memory_cache[key]['timestamp'] < duration:
                return self.memory_cache[key]['data']
        return None
    
    def save(self, key, data):
        """Save data to memory cache."""
        self.memory_cache[key] = {'data': data, 'timestamp': time.time()}
    
    def get_ticker(self, symbol):
        """Get or create yf.Ticker object."""
        if symbol not in self.ticker_cache:
            self.ticker_cache[symbol] = yf.Ticker(symbol)
        return self.ticker_cache[symbol]

# Initialize cache
cache = DataCache()

@lru_cache(maxsize=1)
def get_sp500_companies():
    """Fetches the list of S&P 500 companies from Wikipedia, with caching."""
    if os.path.exists(CACHE_FILE):
        if time.time() - os.path.getmtime(CACHE_FILE) < CACHE_DURATION:
            with open(CACHE_FILE, 'r') as f:
                logger.info("Loading S&P 500 companies from cache")
                return json.load(f)
    
    logger.info("Fetching S&P 500 companies from Wikipedia")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500_symbols = tables[0]['Symbol'].tolist()
        with open(CACHE_FILE, 'w') as f:
            json.dump(sp500_symbols, f)
        return sp500_symbols
    except Exception as e:
        logger.error(f"Error fetching S&P 500 data: {e}")
        # Return empty list or cached data if available
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        return []

def get_stock_data_with_retries(symbols, batch_size=50, max_retries=3, delay=5):
    """Fetch stock data in batches with retry logic and caching."""
    results = {}
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        batch_key = f"batch_{'_'.join(sorted(batch))}"
        
        # Check cache first
        cached_data = cache.get(batch_key)
        if cached_data:
            results.update(cached_data)
            continue
        
        # Retry logic for fetching data
        retries = 0
        while retries < max_retries:
            try:
                # Fetch data for batch
                batch_results = {}
                hist_data = yf.download(batch, period="1d", group_by='ticker')

                # Process each symbol
                for symbol in batch:
                    ticker = cache.get_ticker(symbol)
                    info = ticker.info

                    dividend_yield = info.get('dividendYield', 0)
                    if dividend_yield is None:
                        dividend_yield = 0

                    if symbol in hist_data.columns.levels[0]:
                        current_price = hist_data[symbol]['Close'].iloc[-1]
                    else:
                        current_price = 0

                    batch_results[symbol] = {
                        'dividend_yield': dividend_yield,
                        'price': current_price,
                        'industry': info.get('industry', 'Unknown')
                    }

                # Save to cache
                cache.save(batch_key, batch_results)
                results.update(batch_results)
                break  # Exit retry loop if successful
                
            except Exception as e:
                retries += 1
                logger.warning(f"Error fetching data for {batch}: {e}, retry {retries}/{max_retries}")
                if retries == max_retries:
                    logger.error(f"Max retries reached for {batch}. Skipping.")
                    for symbol in batch:
                        batch_results[symbol] = {
                            'dividend_yield': 0,
                            'price': 0,
                            'industry': 'Unknown'
                        }
                    cache.save(batch_key, batch_results)
                    results.update(batch_results)
                else:
                    time.sleep(delay)  # Wait before retrying

    return results


def get_historical_growth(symbols, period="5y"):
    """Get historical price data and calculate growth rates."""
    growth_results = {}
    
    # Split into manageable batches
    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        cache_key = f"growth_{'_'.join(sorted(batch))}"
        
        # Check cache
        cached_data = cache.get(cache_key)
        if cached_data:
            growth_results.update(cached_data)
            continue
            
        # Fetch historical data
        try:
            hist_data = yf.download(batch, period=period, interval="1mo", group_by='ticker')
            print(f"Fetched data for batch {batch}: {hist_data}")  # Debugging line
            
            batch_results = {}
            for symbol in batch:
                try:
                    if symbol in hist_data.columns.levels[0]:
                        prices = hist_data[symbol]['Close']
                        print(f"Prices for {symbol}: {prices}")  # Debugging line
                        
                        if len(prices) > 1:
                            start_price = prices.iloc[0]
                            end_price = prices.iloc[-1]
                            
                            # Print prices
                            print(f"Start price for {symbol}: {start_price}")
                            print(f"End price for {symbol}: {end_price}")
                            
                            # Calculate growth
                            growth = ((end_price - start_price) / start_price) * 100
                            print(f"Growth for {symbol}: {growth}%")  # Debugging line
                            
                            # Calculate CAGR
                            cagr = (end_price / start_price) ** (1 / 5) - 1 if start_price > 0 else 0
                            print(f"CAGR for {symbol}: {cagr * 100}%")  # Debugging line
                            
                            batch_results[symbol] = {
                                'growth': round(growth, 2),
                                'cagr': round(cagr * 100, 2),
                                'prices': prices.tolist(),
                                'dates': prices.index.strftime('%Y-%m').tolist()
                            }
                        else:
                            batch_results[symbol] = {'growth': 0, 'cagr': 0, 'prices': [], 'dates': []}
                    else:
                        batch_results[symbol] = {'growth': 0, 'cagr': 0, 'prices': [], 'dates': []}
                except Exception as e:
                    logger.warning(f"Error calculating growth for {symbol}: {e}")
                    batch_results[symbol] = {'growth': 0, 'cagr': 0, 'prices': [], 'dates': []}
            
            # Save to cache
            cache.save(cache_key, batch_results)
            growth_results.update(batch_results)
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
    
    return growth_results

def analyze_portfolio(symbols, shares):
    """Comprehensive portfolio analysis."""
    if not symbols or not shares:
        return None
        
    # Get basic stock data
    stock_data = get_stock_data_with_retries(symbols)
    
    # Get historical growth data
    growth_data = get_historical_growth(symbols)
    
    # Calculate portfolio metrics
    results = {}
    total_value = 0
    total_dividend_income = 0
    total_dividend_yield = 0
    dividend_count = 0
    total_growth = 0
    growth_count = 0
    
    # Process each stock
    for symbol in symbols:
        # Skip if no data available
        if symbol not in stock_data or symbol not in growth_data:
            continue
            
        # Get data for calculations
        price = stock_data[symbol]['price']
        dividend_yield = stock_data[symbol]['dividend_yield']
        industry = stock_data[symbol]['industry']
        share_count = shares.get(symbol, 0)
        
        # Calculate metrics
        stock_value = price * share_count
        dividend_income = dividend_yield * stock_value * (1/100)
        
        # Add to totals
        total_value += stock_value
        total_dividend_income += dividend_income
        
        if dividend_yield > 0:
            total_dividend_yield += dividend_yield
            dividend_count += 1
            
        if growth_data[symbol]['growth'] != 0:
            total_growth += growth_data[symbol]['growth']
            growth_count += 1
            
        # Store results
        results[symbol] = {
            'dividend_yield': dividend_yield,
            'dividend_yield_percent': round(dividend_yield, 2),
            'dividend_income': round(dividend_income, 2),
            'stock_value': round(stock_value, 2),
            'industry': industry,
            'growth': growth_data[symbol]['growth'],
            'cagr': growth_data[symbol]['cagr'],
            'projected_value': round(stock_value * (1 + growth_data[symbol]['cagr']/100) ** 5, 2)
        }
    
    # Calculate averages
    avg_dividend_yield = (total_dividend_yield / dividend_count) if dividend_count > 0 else 0
    avg_growth = (total_growth / growth_count) if growth_count > 0 else 0
    
    # Calculate projected portfolio value
    total_projected_value = sum(stock['projected_value'] for stock in results.values())
    projected_growth = ((total_projected_value - total_value) / total_value * 100) if total_value > 0 else 0
    
    # Industry breakdown
    industry_counts = defaultdict(int)
    for symbol, data in results.items():
        industry_counts[data['industry']] += 1
    
    industry_percentages = {industry: (count / len(results) * 100) 
                          for industry, count in industry_counts.items()}
    
    return {
        'stocks': results,
        'metrics': {
            'total_value': round(total_value, 2),
            'total_dividend_income': round(total_dividend_income, 2),
            'average_yield': round(avg_dividend_yield, 2),
            'average_growth': round(avg_growth, 2),
            'total_projected_value': round(total_projected_value, 2),
            'projected_growth': round(projected_growth, 2)
        },
        'industry_breakdown': industry_percentages,
        'growth_data': growth_data
    }

def suggest_stocks(portfolio_data, max_suggestions=8):
    """Suggest additional stocks for portfolio diversification with minimal API calls."""
    current_industries = set(stock['industry'] for stock in portfolio_data['stocks'].values())
    current_symbols = set(portfolio_data['stocks'].keys())

    # Get S&P 500 companies
    sp500_symbols = get_sp500_companies()
    random.shuffle(sp500_symbols)  # Shuffle to avoid bias
    candidate_symbols = sp500_symbols[:100]  

    # Fetch basic data
    all_stocks_data = get_stock_data_with_retries(candidate_symbols, batch_size=50)

    # Filter for missing industries first
    industry_mapping = defaultdict(list)
    for symbol, data in all_stocks_data.items():
        industry = data.get('industry', 'Unknown')

        # 🔥 NEW: Ignore stocks with "Unknown" industry
        if industry == 'Unknown':
            continue

        industry_mapping[industry].append(symbol)

    missing_industries = set(industry_mapping.keys()) - current_industries
    candidates = []

    # Ensure we suggest at least one from missing industries
    selected_industries = set()  # Track selected industries

    for industry in missing_industries:
        for symbol in industry_mapping[industry]:
            if symbol not in current_symbols and industry not in selected_industries:
                candidates.append(symbol)
                selected_industries.add(industry)  # 🔥 NEW: Limit 1 per industry

    # Fallback: Add high-yield/high-growth stocks
    if len(candidates) < max_suggestions:
        extra_candidates = [
            symbol for symbol, data in all_stocks_data.items()
            if symbol not in current_symbols and (data['dividend_yield'] > 0.02 or data['growth'] > 20)
        ]

        for symbol in extra_candidates:
            industry = all_stocks_data[symbol].get('industry', 'Unknown')

            if industry not in selected_industries and industry != 'Unknown':
                candidates.append(symbol)
                selected_industries.add(industry)  # 🔥 NEW: Limit 1 per industry

            if len(candidates) >= max_suggestions:
                break

    # Fetch details for final candidates
    final_candidates = candidates[:max_suggestions]
    detailed_stock_data = get_stock_data_with_retries(final_candidates, batch_size=max_suggestions)

    # Fetch historical growth
    growth_data = get_historical_growth(final_candidates)

    # Construct response
    suggestions = []
    suggested_info = {}

    for symbol in final_candidates:
        data = detailed_stock_data.get(symbol, {})
        growth_value = growth_data.get(symbol, {}).get('growth', 0)

        # 🔥 NEW: Ignore NaN or missing growth values
        if growth_value is None or math.isnan(growth_value):
            continue

        # 🔥 NEW: Ignore negative growth stocks
        if growth_value < 0:
            continue

        # 🔥 NEW: Cap extreme growth values (e.g., max 500% to avoid unrealistic suggestions)
        growth_value = min(growth_value, 500)

        category = 'High Dividend' if data.get('dividend_yield', 0) > 2 else 'High Growth'
        industry = data.get('industry', 'Unknown')

        suggestions.append((symbol, industry, category))
        suggested_info[symbol] = {
            'dividend_yield': round(data.get('dividend_yield', 0), 2),
            'growth_info': f"{round(growth_value, 2)}% over 5 years",
            'category': category
        }

    return suggestions, suggested_info


# Flask application
app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/run_stock_yields', methods=['POST'])
def run_stock_yields():
    """Process stock data and return analysis results."""
    try:
        # Get user input
        symbols = request.form.getlist('symbols[]')
        shares_input = request.form.getlist('shares[]')
        
        # Validate input
        if not symbols or not shares_input or len(symbols) != len(shares_input):
            return render_template('index.html', error_message="Invalid input. Please provide both symbols and shares.")
        
        # Convert shares to integers
        try:
            shares = {symbol: int(share) for symbol, share in zip(symbols, shares_input)}
        except ValueError:
            return render_template('index.html', error_message="Shares must be valid numbers.")
        
        # Analyze portfolio
        portfolio_data = analyze_portfolio(symbols, shares)
        if not portfolio_data:
            return render_template('index.html', error_message="Could not analyze portfolio. Please check your inputs.")
        
        # Get stock suggestions
        suggestions, suggested_info = suggest_stocks(portfolio_data)
        
        # Extract data for template
        metrics = portfolio_data['metrics']
        
        # Prepare chart data
        chart_data = {}
        for symbol, growth_info in portfolio_data['growth_data'].items():
            if growth_info['dates'] and growth_info['prices']:
                chart_data[symbol] = {
                    "dates": growth_info['dates'],
                    "prices": growth_info['prices']
                }
        
        # Return data to template
        return render_template('index.html',
                              dividends=portfolio_data['stocks'],
                              average_yield=metrics['average_yield'],
                              total_dividend_income=metrics['total_dividend_income'],
                              combined_growth=metrics['average_growth'],
                              total_current_value=metrics['total_value'],
                              total_projected_value=metrics['total_projected_value'],
                              combined_projected_growth=metrics['projected_growth'],
                              industry_breakdown=portfolio_data['industry_breakdown'],
                              suggestions=suggestions,
                              suggested_info=suggested_info,
                              chart_data=prepare_chart_data(chart_data))
    
    except Exception as e:
        logger.error(f"Error in run_stock_yields: {e}", exc_info=True)
        return render_template('index.html', error_message=f"An error occurred: {str(e)}")

@app.route('/sp500', methods=['GET'])
def sp500():
    """Fetch the S&P 500 companies."""
    try:
        sp500_symbols = get_sp500_companies()
        return jsonify({
            "success": True,
            "symbols_count": len(sp500_symbols),
            "symbols": sp500_symbols
        })
    except Exception as e:
        logger.error(f"Error in sp500 route: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

def prepare_chart_data(historical_data):
    """Prepare chart data from historical price data."""
    all_dates = set()
    for data in historical_data.values():
        all_dates.update(data['dates'])

    all_dates = sorted(list(all_dates))
    chart_data = {"labels": all_dates, "datasets": []}
    
    # Define colors for chart lines
    colors = [
        'rgba(75, 192, 192, 1)',   # Teal
        'rgba(255, 99, 132, 1)',   # Red
        'rgba(54, 162, 235, 1)',   # Blue
        'rgba(255, 206, 86, 1)',   # Yellow
        'rgba(153, 102, 255, 1)',  # Purple
        'rgba(255, 159, 64, 1)',   # Orange
        'rgba(199, 199, 199, 1)'   # Gray
    ]
    
    for i, symbol in enumerate(historical_data.keys()):
        color_idx = i % len(colors)
        prices_dict = dict(zip(historical_data[symbol]['dates'], historical_data[symbol]['prices']))
        prices_aligned = [prices_dict.get(date, None) for date in all_dates]
        
        chart_data['datasets'].append({
            "label": symbol,
            "data": prices_aligned,
            "borderColor": colors[color_idx],
            "backgroundColor": colors[color_idx].replace('1)', '0.2)'),
            "fill": False
        })

    return chart_data

if __name__ == "__main__":
    # Preload S&P 500 data
    try:
        logger.info("Preloading S&P 500 data...")
        get_sp500_companies()
    except Exception as e:
        logger.error(f"Error during preloading: {e}")
    
    # Start the app
    #app.run(debug=True)
    
    # Use the port that Render provides
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))