import heapq
class RunningPercentile:
    def __init__(self, per):
       self.per = per
       self.left = [] # max heap, so it stores the negative value
       self.right = []

    def len(self): return len(self.left)+len(self.right)

    def add(self, value):
        assert value>0, 'only positive values are supported due to some assert in implementation'
        if len(self.left)==0 or value <= self.leftTop(): # this value should go to left
            self.leftPush(value)
        else:
            self.rightPush(value)

        rMaxLen = int( (1-self.per) * self.len() )
        lMaxLen = self.len() - rMaxLen
        if len(self.left) > lMaxLen:
            self.rightPush(self.leftPop())
        if len(self.right) > rMaxLen:
            self.leftPush(self.rightPop())

    def get(self):
        return self.leftTop()

    def dumpLeft(self): # completely throw away left heap's data to increase the median
        self.left=self.left[:1]

    def rightTop(self): return self.right[0]
    def leftTop(self): assert self.left[0]<0; return -self.left[0]
    def leftPush(self,value): heapq.heappush(self.left, -value)
    def rightPush(self,value):heapq.heappush(self.right, value)
    def leftPop(self): e = heapq.heappop(self.left); assert e<0; return -e;
    def rightPop(self): return heapq.heappop(self.right)

