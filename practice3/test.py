from product.option import *
from pricer.pricer import get_pricing_results
import xlwings as xw 
import pandas as pd

o1 = init_option('简单看涨', 1, 0, 0, 1, 0, 0, 0, 0, 1, 0.3, 0, 0.05)
o2 = init_option('简单看跌', 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.3, 0, 0.05)
o3 = init_option('看涨二元', 1, 0, 0.05, 0, 1, 0, 0, 0, 1, 0.3, 0, 0.05)
o4 = init_option('看跌二元', 1, 0, 0.05, 0, 1, 0, 0, 0, 1, 0.3, 0, 0.05)
o5 = init_option('看涨价差', 1, 0, 0, 1, 1.1, 0, 0, 0, 1, 0.3, 0, 0.05)
o6 = init_option('看跌价差', 1, 0, 0, 0.9, 1, 0, 0, 0, 1, 0.3, 0, 0.05)
o7 = init_option('欧式看涨鲨鱼鳍', 1, 0, 0, 0, 1, 0, 1.1, 0, 1, 0.3, 0, 0.05)
o8 = init_option('欧式看跌鲨鱼鳍', 1, 0, 0, 1, 0, 0.9, 0, 0, 1, 0.3, 0, 0.05)
o9 = init_option('欧式双向鲨鱼鳍', 1, 0, 0, 1, 1, 0.9, 1.1, 0.05, 1, 0.3, 0, 0.05)
o10 = init_option('蝶式', 1, 0, 0, 1, 1, 0.9, 1.1, 0, 1, 0.3, 0, 0.05)
o11 = o10
o12 = init_option('看涨鲨鱼鳍', 1, 0, 0, 0, 1, 0, 1.1, 0, 1, 0.3, 0, 0.05)
o13 = init_option('看跌鲨鱼鳍', 1, 0, 0, 1, 0, 0.9, 0, 0, 1, 0.3, 0, 0.05)
o14 = init_option('双向鲨鱼鳍', 1, 0, 0, 1, 1, 0.8, 1.2, 0.05, 1, 0.3, 0, 0.05)
o15 = init_option('看涨美式触碰', 1, 0, 0, 0, 1.1, 0, 0, 0.05, 1, 0.3, 0, 0.05)
o16 = init_option('看跌美式触碰', 1, 0, 0, 0.9, 1, 0, 0, 0.05, 1, 0.3, 0, 0.05)
o17 = init_option('向下敲入看涨', 1, 0, 0, 0, 0.95, 0.9, 0, 0, 1, 0.3, 0, 0.05)
o18 = init_option('向上敲入看跌', 1, 0, 0, 1.05, 0, 0, 1.1, 0, 1, 0.3, 0, 0.05)

list_options = [o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18]

for i in range(13, 14):
    print(get_pricing_results(list_options[i], greeks=True))

S_endp = (0.3, 2.5)
T_endp = (0.1, 2)





# o_straddle = Straddle(1, 1, Params(0.3, 0, 0.05), 0, 1, 1, 1)
# o_test = DbAmSharkFin(1, 1, Params(0.3, 0, 0.05), 0, 1, 1, 1, 0.9, 1.1, 0.05)

# list2 = [o_straddle, o_test]
# for i in range(len(list2)):
#     print(get_pricing_results(list2[i], greeks=False))


