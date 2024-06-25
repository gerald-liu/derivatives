from io_product.xlsx import XlsxIO, amend_input

io_tool = XlsxIO()

# amend_input('input.xlsx')
# amend_input('output.xlsx')

io_tool.read('input.xlsx')

df_in = io_tool._df_in

portfolio = io_tool.get_portfolio()
io_tool.get_result_products(greeks=True)

io_tool.write_result_products('output.xlsx')

# products = portfolio.get_products()

# io_tool.plot_product('output.xlsx', products[0], (0.3, 2.5), (0.1, 2), greeks=True)

print(portfolio.price())
