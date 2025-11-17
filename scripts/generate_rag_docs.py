"""
Generate RAG documents for enterprise knowledge base
Creates 200 sample documents covering finance, SQL, and code debugging topics
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Enterprise knowledge base documents
FINANCE_DOCS = [
    {
        "title": "Understanding Financial Ratios",
        "content": """Financial ratios are essential tools for analyzing company performance:

1. Liquidity Ratios:
   - Current Ratio = Current Assets / Current Liabilities
   - Quick Ratio = (Current Assets - Inventory) / Current Liabilities
   - Cash Ratio = Cash / Current Liabilities

2. Profitability Ratios:
   - ROE (Return on Equity) = Net Income / Shareholders' Equity
   - ROA (Return on Assets) = Net Income / Total Assets
   - Profit Margin = Net Income / Revenue

3. Leverage Ratios:
   - Debt-to-Equity = Total Debt / Total Equity
   - Interest Coverage = EBIT / Interest Expense

4. Efficiency Ratios:
   - Asset Turnover = Revenue / Total Assets
   - Inventory Turnover = COGS / Average Inventory
"""
    },
    {
        "title": "Time Value of Money Calculations",
        "content": """Core formulas for financial analysis:

Present Value (PV):
PV = FV / (1 + r)^n
Where: FV = Future Value, r = discount rate, n = periods

Future Value (FV):
FV = PV × (1 + r)^n

Net Present Value (NPV):
NPV = Σ [CFt / (1 + r)^t] - Initial Investment

Internal Rate of Return (IRR):
0 = Σ [CFt / (1 + IRR)^t] - Initial Investment

Examples:
- $1,000 invested at 5% for 10 years: FV = $1,628.89
- Project with CF: -$10,000, $3,000, $4,000, $5,000 at 10% discount: NPV = $788.42
"""
    },
    {
        "title": "Stock Valuation Methods",
        "content": """Three primary approaches to stock valuation:

1. Discounted Cash Flow (DCF):
   Value = Σ [FCFt / (1 + WACC)^t] + Terminal Value
   
   Where:
   - FCF = Free Cash Flow
   - WACC = Weighted Average Cost of Capital
   - Terminal Value = FCFn × (1 + g) / (WACC - g)

2. Price-to-Earnings (P/E) Multiples:
   Stock Price = EPS × Industry Average P/E
   
   Trailing P/E = Current Price / Last 12 months EPS
   Forward P/E = Current Price / Forecast EPS

3. Dividend Discount Model (DDM):
   Stock Value = D1 / (r - g)
   
   Where:
   - D1 = Expected dividend next year
   - r = Required rate of return
   - g = Dividend growth rate
"""
    }
]

SQL_DOCS = [
    {
        "title": "Advanced SQL Window Functions",
        "content": """Window functions perform calculations across table rows related to the current row:

1. ROW_NUMBER():
   SELECT employee_id, salary,
          ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as rank
   FROM employees;

2. RANK() and DENSE_RANK():
   RANK() leaves gaps in ranking, DENSE_RANK() doesn't
   SELECT name, score,
          RANK() OVER (ORDER BY score DESC) as rank,
          DENSE_RANK() OVER (ORDER BY score DESC) as dense_rank
   FROM students;

3. LEAD() and LAG():
   SELECT date, revenue,
          LAG(revenue, 1) OVER (ORDER BY date) as prev_revenue,
          LEAD(revenue, 1) OVER (ORDER BY date) as next_revenue
   FROM sales;

4. Moving Averages:
   SELECT date, price,
          AVG(price) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as ma_7
   FROM stock_prices;

5. Cumulative Sum:
   SELECT date, amount,
          SUM(amount) OVER (ORDER BY date) as running_total
   FROM transactions;
"""
    },
    {
        "title": "SQL Query Optimization Techniques",
        "content": """Best practices for optimizing SQL query performance:

1. Index Usage:
   - Create indexes on columns used in WHERE, JOIN, and ORDER BY
   - Use covering indexes to avoid table lookups
   - Monitor index fragmentation

   CREATE INDEX idx_customer_email ON customers(email);
   CREATE INDEX idx_orders_date_customer ON orders(order_date, customer_id);

2. Query Rewriting:
   BAD:  SELECT * FROM orders WHERE YEAR(order_date) = 2023
   GOOD: SELECT * FROM orders WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01'

3. Avoid N+1 Queries:
   Use JOINs instead of multiple queries in loops
   
   BAD:  SELECT * FROM users;
         For each user: SELECT * FROM orders WHERE user_id = ?
   
   GOOD: SELECT u.*, o.*
         FROM users u
         LEFT JOIN orders o ON u.id = o.user_id;

4. Use EXPLAIN:
   EXPLAIN ANALYZE SELECT * FROM large_table WHERE indexed_col = 'value';

5. Batch Operations:
   INSERT INTO table VALUES (1, 'a'), (2, 'b'), (3, 'c');  -- Better than 3 separate INSERTs
"""
    },
    {
        "title": "Common Table Expressions (CTE) Patterns",
        "content": """CTEs improve query readability and enable recursive queries:

1. Basic CTE:
   WITH sales_summary AS (
       SELECT product_id, SUM(quantity) as total_qty, SUM(amount) as total_amount
       FROM sales
       WHERE sale_date >= '2023-01-01'
       GROUP BY product_id
   )
   SELECT p.name, s.total_qty, s.total_amount
   FROM products p
   JOIN sales_summary s ON p.id = s.product_id
   ORDER BY s.total_amount DESC;

2. Multiple CTEs:
   WITH 
   revenue AS (SELECT customer_id, SUM(amount) as total FROM orders GROUP BY customer_id),
   costs AS (SELECT customer_id, SUM(cost) as total FROM expenses GROUP BY customer_id)
   SELECT r.customer_id, r.total - COALESCE(c.total, 0) as profit
   FROM revenue r
   LEFT JOIN costs c ON r.customer_id = c.customer_id;

3. Recursive CTE (Employee Hierarchy):
   WITH RECURSIVE employee_tree AS (
       SELECT id, name, manager_id, 0 as level
       FROM employees
       WHERE manager_id IS NULL
       UNION ALL
       SELECT e.id, e.name, e.manager_id, et.level + 1
       FROM employees e
       JOIN employee_tree et ON e.manager_id = et.id
   )
   SELECT * FROM employee_tree ORDER BY level, name;
"""
    }
]

PYTHON_DOCS = [
    {
        "title": "Python Performance Optimization",
        "content": """Techniques to optimize Python code performance:

1. Use Built-in Functions and Libraries:
   # Slow
   result = []
   for i in range(1000000):
       result.append(i * 2)
   
   # Fast
   result = [i * 2 for i in range(1000000)]
   
   # Fastest
   import numpy as np
   result = np.arange(1000000) * 2

2. Avoid Global Variables:
   # Slow
   global_list = []
   def process():
       for i in range(1000):
           global_list.append(i)
   
   # Fast
   def process():
       local_list = []
       for i in range(1000):
           local_list.append(i)
       return local_list

3. Use Generators for Large Datasets:
   # Memory intensive
   def get_numbers():
       return [i for i in range(1000000)]
   
   # Memory efficient
   def get_numbers():
       for i in range(1000000):
           yield i

4. Profile Your Code:
   import cProfile
   cProfile.run('your_function()')
   
   # Or use line_profiler
   @profile
   def your_function():
       pass

5. Multiprocessing for CPU-bound Tasks:
   from multiprocessing import Pool
   
   def process_item(item):
       return item * 2
   
   with Pool(4) as p:
       results = p.map(process_item, range(1000000))
"""
    },
    {
        "title": "Common Python Debugging Patterns",
        "content": """Essential debugging techniques for Python:

1. Print Debugging with Context:
   def complex_function(data):
       print(f"DEBUG: Input type={type(data)}, len={len(data)}, first={data[0]}")
       result = process(data)
       print(f"DEBUG: Result type={type(result)}, value={result}")
       return result

2. Using pdb Debugger:
   import pdb
   
   def problematic_function():
       x = 10
       y = 0
       pdb.set_trace()  # Execution stops here
       result = x / y   # You can inspect variables
       return result

3. Logging Best Practices:
   import logging
   
   logging.basicConfig(level=logging.DEBUG, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   logger = logging.getLogger(__name__)
   
   def process_data(data):
       logger.info(f"Processing {len(data)} items")
       try:
           result = risky_operation(data)
           logger.debug(f"Operation result: {result}")
       except Exception as e:
           logger.error(f"Error processing data: {e}", exc_info=True)
           raise

4. Assert for Assumptions:
   def calculate_average(numbers):
       assert len(numbers) > 0, "Cannot calculate average of empty list"
       assert all(isinstance(n, (int, float)) for n in numbers), "All items must be numbers"
       return sum(numbers) / len(numbers)

5. Context Managers for Resource Management:
   from contextlib import contextmanager
   
   @contextmanager
   def debug_timer(name):
       import time
       start = time.time()
       try:
           yield
       finally:
           print(f"{name} took {time.time() - start:.2f}s")
   
   with debug_timer("Data processing"):
       process_large_dataset()
"""
    },
    {
        "title": "Python Async/Await Patterns",
        "content": """Modern asynchronous programming in Python:

1. Basic Async Function:
   import asyncio
   
   async def fetch_data(url):
       await asyncio.sleep(1)  # Simulates I/O
       return f"Data from {url}"
   
   async def main():
       result = await fetch_data("https://api.example.com")
       print(result)
   
   asyncio.run(main())

2. Concurrent Execution:
   async def main():
       tasks = [
           fetch_data("url1"),
           fetch_data("url2"),
           fetch_data("url3")
       ]
       results = await asyncio.gather(*tasks)
       print(results)

3. Async Context Managers:
   class AsyncDatabaseConnection:
       async def __aenter__(self):
           print("Connecting to database...")
           await asyncio.sleep(1)
           return self
       
       async def __aexit__(self, exc_type, exc, tb):
           print("Closing database connection...")
           await asyncio.sleep(1)
       
       async def query(self, sql):
           await asyncio.sleep(0.5)
           return f"Results for: {sql}"
   
   async def main():
       async with AsyncDatabaseConnection() as conn:
           result = await conn.query("SELECT * FROM users")

4. Async Iterators:
   class AsyncRange:
       def __init__(self, n):
           self.n = n
           self.i = 0
       
       def __aiter__(self):
           return self
       
       async def __anext__(self):
           if self.i < self.n:
               await asyncio.sleep(0.1)
               self.i += 1
               return self.i
           raise StopAsyncIteration
   
   async def main():
       async for i in AsyncRange(5):
           print(i)

5. Error Handling in Async:
   async def risky_async_operation():
       try:
           result = await potentially_failing_operation()
           return result
       except ConnectionError as e:
           logger.error(f"Connection failed: {e}")
           await asyncio.sleep(5)  # Wait before retry
           return await risky_async_operation()  # Retry
       except Exception as e:
           logger.exception("Unexpected error")
           raise
"""
    }
]


def generate_rag_documents(output_dir: str = "data/rag_documents", num_docs: int = 200):
    """Generate enterprise RAG documents"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {num_docs} RAG documents...")
    
    # Distribute across categories
    finance_count = int(num_docs * 0.4)  # 80 docs
    sql_count = int(num_docs * 0.3)      # 60 docs
    python_count = num_docs - finance_count - sql_count  # 60 docs
    
    all_docs = []
    doc_id = 1
    
    # Generate finance documents
    for i in range(finance_count):
        base_doc = random.choice(FINANCE_DOCS)
        doc = {
            "id": f"fin_{doc_id:03d}",
            "title": f"{base_doc['title']} - Part {(i % 3) + 1}",
            "content": base_doc['content'],
            "category": "finance",
            "metadata": {
                "source": "enterprise_knowledge_base",
                "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                "tags": ["finance", "accounting", "analysis"]
            }
        }
        all_docs.append(doc)
        
        # Save individual document
        with open(output_path / f"{doc['id']}.txt", 'w', encoding='utf-8') as f:
            f.write(f"Title: {doc['title']}\n")
            f.write(f"Category: {doc['category']}\n")
            f.write(f"Tags: {', '.join(doc['metadata']['tags'])}\n\n")
            f.write(doc['content'])
        
        doc_id += 1
    
    # Generate SQL documents
    for i in range(sql_count):
        base_doc = random.choice(SQL_DOCS)
        doc = {
            "id": f"sql_{doc_id:03d}",
            "title": f"{base_doc['title']} - Example {(i % 3) + 1}",
            "content": base_doc['content'],
            "category": "sql",
            "metadata": {
                "source": "enterprise_knowledge_base",
                "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                "tags": ["sql", "database", "query-optimization"]
            }
        }
        all_docs.append(doc)
        
        with open(output_path / f"{doc['id']}.txt", 'w', encoding='utf-8') as f:
            f.write(f"Title: {doc['title']}\n")
            f.write(f"Category: {doc['category']}\n")
            f.write(f"Tags: {', '.join(doc['metadata']['tags'])}\n\n")
            f.write(doc['content'])
        
        doc_id += 1
    
    # Generate Python documents
    for i in range(python_count):
        base_doc = random.choice(PYTHON_DOCS)
        doc = {
            "id": f"py_{doc_id:03d}",
            "title": f"{base_doc['title']} - Guide {(i % 3) + 1}",
            "content": base_doc['content'],
            "category": "python",
            "metadata": {
                "source": "enterprise_knowledge_base",
                "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                "tags": ["python", "coding", "debugging"]
            }
        }
        all_docs.append(doc)
        
        with open(output_path / f"{doc['id']}.txt", 'w', encoding='utf-8') as f:
            f.write(f"Title: {doc['title']}\n")
            f.write(f"Category: {doc['category']}\n")
            f.write(f"Tags: {', '.join(doc['metadata']['tags'])}\n\n")
            f.write(doc['content'])
        
        doc_id += 1
    
    # Save metadata
    with open(output_path / "documents_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Generated {len(all_docs)} documents:")
    logger.info(f"   - Finance: {finance_count}")
    logger.info(f"   - SQL: {sql_count}")
    logger.info(f"   - Python: {python_count}")
    logger.info(f"📁 Saved to {output_path}")


if __name__ == "__main__":
    generate_rag_documents()
