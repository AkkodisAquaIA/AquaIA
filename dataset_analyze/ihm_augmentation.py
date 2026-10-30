import multiprocessing

from ihm.interface_augmentation  import InterfaceAugmentation


def main():
    application = InterfaceAugmentation()
    application.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
