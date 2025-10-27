class Router:
    def __init__(self, frames):
        self.frames = frames

    def show(self, frame_name):
        for name, frame in self.frames.items():
            if name == frame_name:
                frame.tkraise()
