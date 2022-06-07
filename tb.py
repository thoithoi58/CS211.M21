from tensorboard import program

tracking_address = 'logs/' # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")