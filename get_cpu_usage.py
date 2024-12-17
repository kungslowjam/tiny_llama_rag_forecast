import psutil

def get_cpu_usage():
    try:
        # ดึงเปอร์เซ็นต์การใช้งาน CPU
        cpu_usage = psutil.cpu_percent(interval=1)  # interval=1 หมายถึงรอ 1 วินาทีก่อนวัดค่า
        return f"CPU Usage: {cpu_usage}%"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print(get_cpu_usage())
