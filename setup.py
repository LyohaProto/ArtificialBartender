import urllib.request

BASE_URL = 'https://s3-ap-southeast-2.amazonaws.com/artificial-bartender'
bin_files = ['age_net.caffemodel', 'gender_net.caffemodel']

for bin_file in bin_files:
	s3_file = f'{BASE_URL}/{bin_file}'
	with urllib.request.urlopen(s3_file) as response:
		print(f"Downloading {s3_file}...")
		content = response.read()
		local_filename = f'age_gender_models/{bin_file}'
		with open(local_filename, 'wb') as local_file:
			print(f"Writing local file {local_filename}")
			local_file.write(content)

print("Done!")