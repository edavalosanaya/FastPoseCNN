import catalyst.core.callback

#-------------------------------------------------------------------------------

class MyCustomCallback(catalyst.core.callback.Callback):
    def __init__(self):
        super().__init__(order=120)

        print("Visualize mask")