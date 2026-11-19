"""Fallback runner when pytest is not installed: python simple_test_runner.py"""
import glob, importlib, sys, traceback

sys.path.insert(0, ".")
passed = failed = 0
for path in sorted(glob.glob("test_*.py")):
    mod = importlib.import_module(path[:-3])
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        try:
            getattr(mod, name)()
            passed += 1
            print("PASS", path, name)
        except Exception:
            failed += 1
            print("FAIL", path, name)
            traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
