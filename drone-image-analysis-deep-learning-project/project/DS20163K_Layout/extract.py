'''
	File name: extract.py
	Program Description: A program that extracts RGB values of every plot and mapping those values with the corresponding feature in the shapefile. 
	Date: 22 July 2018
	Author: Loria Roie Grace N. Malingan 
'''

import ogr, os, sys
import geopandas as gpd
from segment import * 
from ranges import *

# Read shapefile
fn = "./DS20163K_Layout/G3kPlotsbuff.shp"
driver = ogr.GetDriverByName('ESRI Shapefile')
dataSource = driver.Open(fn, 0)

# Get Layer
print 'The shapefile contains ', dataSource.GetLayerCount(), ' Layer/s'
layer = dataSource.GetLayer(0)

# Get Features
print layer.GetName(), ' contains ', layer.GetFeatureCount(), ' features'

pixel_list = list(img.getdata()) # get image data (rgb values) from the drone image
pixel_dict = {}	# declare an empty dictionary
i = 0
width, height = img.size # get the height and width of the drone image
# traverse through the image to get rgb values of each pixel
for y in range(height):
	for x in range(width):
		pixel_dict[(x,y)] = pixel_list[i] # add a key:value to the dictionary where key=(x,y)-coordinates of the pixel and value is the rgb values
		i+=1 


# Access the properties of labeled regions (segments from the slic algo)
regions = measure.regionprops(segments_slic, intensity_image=None, cache=True)
feature = layer.GetFeature(0) # get the first feature in the shapefile
df = gpd.GeoDataFrame(feature.items(), index=[1]) # create a dataframe and add the feature from the shapefile
# traverse through the shapefile
for i in range(2,layer.GetFeatureCount()):
	feature = layer.GetFeature(i) # get the feature at index i (start at 2)
	df_temp = gpd.GeoDataFrame(feature.items(), index=[i]) # create a temporary dataframe
	row = feature.GetField('Row') # access plot row
	col = feature.GetField('Col') # access plot column
	
	# set the variables to zero for this iteration
	r_total = 0
	g_total = 0
	b_total = 0
	sp = 0
	
	# check if the plot is located at the top field
	if col < 9:
		for plot in regions:
			# check if the plot is within the range of the x-axis of the row
			if plot.centroid[0] in range(top_field_x[row-1][0],top_field_x[row-1][1]):
				if plot.centroid[1] in range(top_field_y[col-9][0],top_field_y[col-9][1]):
					sp = plot.label-1 # decrement by 1 because label starts at 1 
					print "hi"
					break

	# otherwise, the plot is located at the bottom field
	else:
		for plot in regions:
			# check if the plot is within the range of the x-axis of the row
			if plot.centroid[0] in range(bottom_field_x[row-1][0],bottom_field_x[row-1][1]):
				if plot.centroid[1] in range(bottom_field_y[col-1][0],bottom_field_y[col-1][1]):
					sp = plot.label-1 # decrement by 1 because label starts at 1 
					print "hello"
					break

	# After identifying the correct plot, access every pixel in the plot
	total_pixels = len(superpixel_list[sp])
	for i in range(total_pixels):
		x_coord = superpixel_list[sp][0][i]		# get x-coordinate 
		y_coord = superpixel_list[sp][1][i]		# get y-coordinate
		r,g,b =  pixel_dict[(x_coord,y_coord)] 	# get the rgb values of the pixel at coordinate(x,y)
		#print pixel_dict[(x_coord,y_coord)]	
		r_total += r
		g_total += g
		b_total += b
	# compute for the average R,G,B values of each plot
	r_avg = r_total/total_pixels
	g_avg = g_total/total_pixels
	b_avg = b_total/total_pixels
	
	# add R,G,B columns to the temporary dataframe 
	df_temp['R'] = np.where(df_temp['Row'] == row, r_avg, 'NA')
	df_temp['G'] = np.where(df_temp['Row'] == row, g_avg, 'NA')
	df_temp['B'] = np.where(df_temp['Row'] == row, b_avg, 'NA')

	df = df.append(df_temp, sort=True) # add the temporary dataframe to the main dataframe
	
		
# rearrange columns
df = df[['Id', 'Field', 'Plot','Row','Col','Elevation','Entry','Treatment', 'R', 'G', 'B']]
# convert the dataframe into a csv file
df.to_csv("../csv-files/output.csv", sep=",")


#end 
