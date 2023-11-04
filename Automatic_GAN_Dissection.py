import subprocess

# Read the list of words from a text file
with open('word_list.txt', 'r') as file:
    word_list = [line.strip() for line in file]

# Define the path to the Automatic_GAN_Dissection.py script
gan_dissection_script = 'GAN_Dissection.py'

# Loop through each word in the list and execute the script
for attribute in word_list:
    print(f"Processing attribute: {attribute}")

    # Run the Automatic_GAN_Dissection.py script with the current attribute as an argument
    subprocess.run(['python', gan_dissection_script, '--attribute', attribute])

    print(f"Attribute '{attribute}' processed successfully")

    # You can add any additional code here to handle the results or perform other tasks

print("All attributes processed")
