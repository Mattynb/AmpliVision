import unittest


if __name__ == "__main__":
    print("Running tests...", "-"*30, sep="\n")
    
    # Discover and load all the tests from the 'test' folder
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=".", pattern='test_*.py')

    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)