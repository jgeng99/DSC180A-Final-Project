import time

def log_curr(parent, cur_dir, text, to_train=True):
    if to_train:
        path = parent + cur_dir + "_train_log.txt"
    else:
        path = parent + cur_dir + "_result_log.txt"
    with open(path, "a") as f:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S:%", t)
        towrite = f"{current_time} {text}\n"
        f.write(towrite)