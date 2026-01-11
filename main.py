import subprocess
import sys

def run(script_name):
    print(f"\n>>> Uruchamiam: {script_name}\n")
    subprocess.run([sys.executable, script_name])


def main_menu():
    print("=" * 50)
    print("PROJEKT: KLASTERYZACJA – MENU GŁÓWNE")
    print("=" * 50)
    print("1. Dane syntetyczne – algorytmy + metryki + wykresy")
    print("2. Dane syntetyczne – stabilność DBSCAN (eps)")
    print("3. Dane rzeczywiste (miasta 2020) – metryki jakości")
    print("4. Dane rzeczywiste (miasta 2020) – stabilność")
    print("5. Dane rzeczywiste – DBSCAN eps (stabilność)")
    print("6. Dane rzeczywiste – metryki + stabilność")
    print("0. Wyjście")
    print("=" * 50)

    choice = input("Wybierz opcję: ")
    return choice


if __name__ == "__main__":
    while True:
        choice = main_menu()

        if choice == "1":
            run("experiment_synthetic_basic.py")

        elif choice == "2":
            run("experiment_synthetic_dbscan_eps.py")

        elif choice == "3":
            run("experiment_real_metrics.py")

        elif choice == "4":
            run("experiment_real_stability.py")

        elif choice == "5":
            run("experiment_real_dbscan_eps.py")

        elif choice == "6":
            run("experiment_real_metrics_and_stability.py")

        elif choice == "0":
            print("Zamykanie programu.")
            break

        else:
            print("Nieprawidłowy wybór, spróbuj ponownie.")
