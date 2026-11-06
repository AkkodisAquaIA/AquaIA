import multiprocessing

from ihm.info_imgages  import InterfaceInfoImg


def main():
    application = InterfaceInfoImg()
    application.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
