import os
import shutil
import glob


newpathT1 = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\train\krank' 
newpathT2 = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\train\gesund' 

newpathV1 = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\validate\krank' 
newpathV2 = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\validate\gesund' 

newpathTest1 = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\test\krank' 
newpathTest2 = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\test\gesund' 


if not os.path.exists(newpathT1):
    os.makedirs(newpathT1)

if not os.path.exists(newpathT2):
    os.makedirs(newpathT2)

if not os.path.exists(newpathV1):
    os.makedirs(newpathV1)

if not os.path.exists(newpathV2):
    os.makedirs(newpathV2)

if not os.path.exists(newpathTest1):
    os.makedirs(newpathTest1)

if not os.path.exists(newpathTest2):
    os.makedirs(newpathTest2)



src = r'C:\Users\erics\source\repos\BrustkrebsMachineLearning\BrustkrebsMachineLearning\Datasets\origin'

print(src)

counter = 0
i = 0
extension = '0.png'

for folders, subfolders, filenames in os.walk(src):
    for filename in filenames:
        filepath = folders + os.sep + filename
        if(i == 0):
            if filepath.endswith("0.png"):
                shutil.copy(os.path.join(folders, filename), newpathV2)
            else:
                shutil.copy(os.path.join(folders, filename), newpathV1)
        elif i == 1 or i == 2:
            if filepath.endswith("0.png"):
                shutil.copy(os.path.join(folders, filename), newpathTest2)
            else:
                shutil.copy(os.path.join(folders, filename), newpathTest1)
        else:
            if filepath.endswith("0.png"):
                shutil.copy(os.path.join(folders, filename), newpathT2)
            else:
                shutil.copy(os.path.join(folders, filename), newpathT1)

        if(i == 9):
            i = 0
        else:
            i+=1

        counter += 1
        print(counter)