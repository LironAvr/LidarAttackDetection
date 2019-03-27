# Define the input sample path
You need to provide path to the ldscan_current.pkl which is generated from the Botvac application at runtime. Modify this variable in the web.py file:
````
FILE_LOCATION = '../Botvac-Control/ldscan_current.pkl' # For example
````
Due to the experiment which was having the permanent error angles so we just ignored them. You can modify this by changing this array:
````
IGNORE_ANGLES = [25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                 85, 86, 87, 88, 89, 90, 91, 97, 98, 122, 126, 127, 128, 129, 130, 131, 132, 133, 136, 137, 139, 140,
                 141, 142, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
                 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183,
                 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
                 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                 224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
                 246, 247, 248, 264, 279, 280, 281, 296, 297, 298, 299, 306, 307, 308, 309, 310, 311, 312, 315, 325,
                 326, 327, 328, 329, 330, 331, 338, 339, 340, 341, 55, 66, 135, 138, 143, 146, 178, 227, 235, 278, 295,
                 249, 134, 253, 42, 37, 93, 305, 290, 294, 266, 32, 300, 332, 314, 283, 68, 43, 343, 0, 342, 251, 44,
                 49, 2, 34, 282, 333, 268, 301, 67, 33, 250, 92, 302, 74, 334, 100, 313, 36, 45, 293, 324, 3, 95, 317,
                 4, 319, 99, 124, 35, 303, 316, 336, 96, 101, 255, 286, 304] # for example
````
# Run program
Run the web.py file (Python3), wait for a while and access to http://127.0.0.1:5000, the web page will appear the result of the state of the Botvac (under attack or not)
# Modify the training algorithm
Feel free to make it better, the program is using SVM RBF kernel for classifying.