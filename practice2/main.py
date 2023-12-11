from io_product.xlsx import XlsxIO

product_type = 'sharkfin'
input_xlsx_filename = 'input.xlsx'
input_sheetname = 'Sheet1'
output_xlsx_filename = 'output.xlsx'
output_sheetname = 'Sheet1'
output_img_filename = 'output.png'

io_tool = XlsxIO(product_type)
io_tool.read(input_xlsx_filename, sheetname=input_sheetname)

results = io_tool.get_results()
print('Closed-form solution:\n', results)

results_mc = io_tool.get_results(method='monte-carlo', inplace=False, price_only=True)
print('\nMonte-Carlo solution (price only):\n', results_mc)

df_error = (results_mc - results[results_mc.columns]) * 10000
print('\nError between MC & CF results (bp):\n', df_error)
      
io_tool.write_results(output_xlsx_filename, sheetname=output_sheetname)
io_tool.plot(output_img_filename, S_endp=(0.3, 2.5), T_endp=(0.1, 2), N=100)
io_tool.write_image(output_img_filename, output_xlsx_filename, sheetname=output_sheetname)
