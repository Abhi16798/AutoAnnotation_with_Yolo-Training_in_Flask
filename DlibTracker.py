import dlib

############################   Dlib Tracking   ############################
class DlibTracker():
    def __init__(self):
        self.tracker = dlib.correlation_tracker()
        self.bInit = False
        self.cnt = 0
		
    def __del__(self):
        self.bInit = False

    def reset(self, x, y, w, h):
        self.rect = dlib.rectangle(x, y, x+w, y+h)
        self.bInit = False
        self.cnt = 0
                
    def update(self, rgb):
        if self.bInit:
            self.tracker.update(rgb)
            self.cnt +=1
        else:            
            self.tracker.start_track(rgb, self.rect)
            self.bInit = True
            self.cnt +=1

    def getPos(self):
        return self.tracker.get_position()
        
