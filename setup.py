import urllib.request
import math
import sys

BASE_URL = 'https://s3-ap-southeast-2.amazonaws.com/artificial-bartender'
bin_files = ['age_net.caffemodel', 'gender_net.caffemodel']

for bin_file in bin_files:
	s3_file = f'{BASE_URL}/{bin_file}'
	local_filename = f'age_gender_models/{bin_file}'
	sys.stdout.write(f"Downloading {s3_file}... ")
	with urllib.request.urlopen(s3_file) as response, open(local_filename, 'wb') as local_file:
		total_bytes = response.length
		chunk = 100 * 1024  # download 100kb chunks
		content = response.read(chunk)
		counter = 1
		total_chunks = math.ceil(total_bytes / chunk)
		while content:
			local_file.write(content)
			percentage = round(counter / total_chunks * 100, 1)
			percentage_len = len(str(percentage)) + 10  # 10 additional characters for text
			content = response.read(100 * 1024)
			counter += 1
			sys.stdout.write(f"{percentage}% complete")
			sys.stdout.flush()
			sys.stdout.write("\b" * percentage_len)
		sys.stdout.write("Completed.     \n")

print("Done!")