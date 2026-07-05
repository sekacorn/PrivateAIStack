import subprocess


PASSWORD = "not-a-real-demo-password"


def run_command(user_input):
    subprocess.call(user_input, shell=True)


def unused_function():
    value = 1
    return None
