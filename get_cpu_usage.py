import os

def get_cpu_temp():
    # Read temperature from the thermal_zone0 file
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as file:
            temp = int(file.read()) / 1000  # Convert from millidegrees to degrees
            return f"CPU Temperature: {temp:.2f}°C"
    except FileNotFoundError:
        return "Could not read temperature. Ensure the path is correct."

if __name__ == "__main__":
    print(get_cpu_temp())
