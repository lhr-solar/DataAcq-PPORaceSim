def test():
#     t = Track("trackDynamic")
#     t.setForks("left", "left", "left")

#     while t.getNext(t.getCurr()) != "S0":
#         print(t.getCurr())
#         t.goNext()
#     print(t.getCurr())
#     t.goNext()

#     print("shortest track length: " + str(t.getDistance("S0", "S0"))) #
#     t.pitNext()
#     print("above length with pit lap: " + str(t.getDistance("S0", "S0"))) # this inherently 2 laps to make it to the finish line after a pit
#     t.setForks("right", "right", "right")
#     print("longest track length: " + str(t.getDistance("S0", "S0")/5280))
#     # t.generateMinimap()
#     t.interpolateMinimap()

# test()