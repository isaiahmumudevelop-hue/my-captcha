import os

def check_code():
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found! Make sure you are in the ~/real_rechapcha folder.")
        return

    with open("app.py", "r") as f:
        code = f.read()

    checks = {
        "Green Checkmark (SVG)": '<svg id="cb-checkmark"',
        "Countdown Timer HTML": 'id="timer-display"',
        "Timer JavaScript": 'function startTimer()',
        "Dark Mode CSS": 'body.dark-mode',
        "Dark Mode JavaScript": 'function toggleTheme()',
        "Admin Stats Dictionary": 'stats = {',
        "Admin Dashboard Route": "@app.route('/admin')",
        "Stats tracking in verify": 'stats["success"] += 1',
        "Stats tracking in index": 'stats["total_loads"] += 1'
    }

    print("🔍 Checking your app.py for all the new features...\n")
    all_good = True
    for name, text in checks.items():
        if text in code:
            print(f"✅ {name}: Found")
        else:
            print(f"❌ {name}: MISSING! You need to add this.")
            all_good = False

    if all_good:
        print("\n🎉 Perfect! All features are in the code.")
    else:
        print("\n⚠️ Some features are missing. Check the ones marked with ❌.")

if __name__ == "__main__":
    check_code()
