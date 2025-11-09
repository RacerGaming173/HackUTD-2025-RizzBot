import subprocess
import sys


def display_menu():
    print("Select RizzBot mode:")
    print("1. Regular RizzBot")
    print("2. Low-latency RizzBot")
    return input("Enter your choice: ").strip()


def display_voice_menu():
    print("\nChoose a voice for low-latency RizzBot:")
    print("1. Leda")
    print("2. Default")
    return input("Enter your voice choice: ").strip()


def run_command(command, label=None):
    label = label or " ".join(command)

    try:
        print(f"\n--- Executing: {label} ---")
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"--- {label} finished with return code {result.returncode} ---")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {label} failed with return code {e.returncode}")
    except FileNotFoundError:
        print("ERROR: Python interpreter not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    script_choice = display_menu()

    if script_choice == "1":
        run_command([sys.executable, "src/tts.py"], label="Regular RizzBot")
    elif script_choice == "2":
        voice_choice = display_voice_menu()

        base_command = [
            sys.executable,
            "src/live-audio.py",
            "--mode",
            "speech",
            "--system-prompt-file",
            "src/system_prompt.txt",
            "--input-device",
            "1",
            "--output-device",
            "11",
            "--sample-rate",
            "44100",
            "--clip-seconds",
            "0.2",
            "--playback-chunk-seconds",
            "3",
        ]

        if voice_choice == "1":
            run_command(
                base_command + ["--voice", "Leda"],
                label="Low-latency RizzBot (Voice: Leda)",
            )
        elif voice_choice == "2":
            run_command(base_command, label="Low-latency RizzBot (Default Voice)")
        else:
            print("Invalid voice choice. Exiting.")
    else:
        print("Invalid selection. Exiting.")
