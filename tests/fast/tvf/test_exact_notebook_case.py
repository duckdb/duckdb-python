#!/usr/bin/env python3
"""
Test the EXACT code from the user's notebook in a real IPython environment.
This should reproduce the hanging issue if it still exists.
"""

import sys
import time

# Set up REAL IPython environment (not just simulated)
def setup_real_ipython():
    """Start a real IPython kernel/shell to match VS Code Jupyter behavior"""
    try:
        import IPython
        from IPython.terminal.interactiveshell import TerminalInteractiveShell
        
        # Start IPython shell instance 
        shell = TerminalInteractiveShell.instance()
        
        # Make sure get_ipython() returns the shell
        IPython.get_ipython = lambda: shell
        
        # Start the shell to activate all IPython features
        shell.init_readline()
        
        print(f"✅ Real IPython shell started: {type(shell)}")
        return shell
    except Exception as e:
        print(f"❌ Failed to start real IPython: {e}")
        return None

def run_exact_user_code():
    """Run the exact code the user provided from their notebook"""
    import duckdb
    
    print("🧪 Running EXACT user code from notebook...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # EXACT CODE FROM USER - no modifications
        def generate_names(count: int = 500):
            """Generate a list of sample names."""
            names = ["Alice1", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]

            for i in range(count):
                yield (names[i % len(names)], i)
            #return [(names[i % len(names)], i) for i in range(count)]

        with duckdb.connect() as conn:
            # Register the TVF with schema definition
            schema = [["name", "VARCHAR"], ["id", "INT"]]

            tvf = conn.create_table_function(
                name="generate_names",
                callable=generate_names,
                parameters=None,  # Will infer from function signature
                schema=schema,
                return_type="records",  # Return list of tuples
            )

            # Use the TVF in a query
            n = 100000  # User's exact test case
            print(f"⚡ Starting query with n={n} at {time.time() - start_time:.1f}s")
            result = conn.sql(f"set threads = 1;SELECT * FROM generate_names({n})").df()
            
            elapsed = time.time() - start_time
            print(f"✅ Query completed at {elapsed:.1f}s")
            
            assert len(result) == n, len(result)
            print(f"✅ Assertion passed - got {len(result)} rows")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"💥 ERROR at {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def main():
    print("🔍 Testing EXACT user notebook code in real IPython environment")
    print("This should reproduce the hang if it still exists")
    
    # Start real IPython environment
    shell = setup_real_ipython()
    if not shell:
        print("⚠️  Cannot test without IPython - install with: pip install ipython")
        return
    
    # Run the exact user code 
    print("🚀 Executing user's exact notebook code...")
    success = run_exact_user_code()
    
    if success:
        print("🎉 SUCCESS - No hang detected!")
        print("The fix appears to be working correctly")
    else:
        print("💥 FAILED - Issue reproduced")
        print("The hang still exists in real IPython environment")

if __name__ == "__main__":
    main()