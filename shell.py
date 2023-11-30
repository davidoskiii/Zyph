import basic
import readline

import sys
import signal, os

try:
    basic.run_script(sys.argv[1])
except IndexError:
    def handler(signum, frame):
        print("\nUse 'quit()' to exit\n> ", end="")

    # Set the signal handler
    signal.signal(signal.SIGINT, handler)

    print("Read the documentation for help")

    while True:
        text = input("> ")
        if text.strip() == "": continue
        result, error = basic.run('<stdin>', text)

        if error:
            print(error.as_string())
        elif result:
            if len(result.elements) == 1:
                print(repr(result.elements[0]))
            else:
                print(repr(result))