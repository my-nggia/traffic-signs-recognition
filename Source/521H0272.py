import cv2
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd

def read_img(img_path):
    return cv2.imread(img_path)

def gray_img(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def detect_img(img, img_name, show, export):
    """
    * Đặc điểm của biển báo cấm (thông dụng): 
    - hình tròn
    - nền trắng
    - viền ngoài màu đỏ
    - trong hình tròn có dấu gạch chéo màu đỏ hoặc hình vẽ bên trong màu đen
    
    *  1 số biển cấm đặc biệt (không áp dụng trong bài này)
    """
    try: 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # đỏ hoặc cam đỏ
        mask_r1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        # màu xanh dương hoặc màu lam
        mask_r2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        
        mask_r = cv2.bitwise_or(mask_r1, mask_r2)
        target = cv2.bitwise_and(img, img, mask=mask_r)
        gblur = cv2.GaussianBlur(mask_r, (9, 9), 0)
        edge_img = cv2.Canny(gblur, 30, 150)
        
        img2 = img.copy()
        cnts, hierarchy = cv2.findContours(edge_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img2, cnts, -1, (0, 255, 0), 2)
        
        img2 = img.copy()
        model = load_model('model/model.h5')
        try:
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if(area < 800):
                    continue
                # ellipse = cv2.fitEllipse(cnt)
                # cv2.ellipse(img2, ellipse, (0, 255, 0), 2)
                
                x, y, w, h = cv2.boundingRect(cnt)
                a, b, c, d = x, y, w, h
                
                # Random color for multiple objects
                B = random.randint(0, 255)
                G = random.randint(0, 255)
                R = random.randint(0, 255)
                
                cv2.rectangle(img2, (x, y), (x+w, y+h), (B, G, R), 2)
                
                crop = img2[b:b+d, a:a+c]
                specific_str = str(random.randint(1, 1000))
                crop_name = export_crop_img(crop, img_name, specific_str)
                
                data = []
                image_from_array = Image.fromarray(crop, 'RGB')
                crop = image_from_array.resize((30, 30))
                data.append(np.array(crop))
                X_test = np.array(data)
                X_test = X_test.astype('float32')/255
                prediction = model.predict(X_test)
                classes = np.argmax(prediction, axis=1)
                str_classes = get_class_name(classes, crop_name)
                cv2.putText(img2, str(str_classes), (x+w+5, y), cv2.FONT_HERSHEY_SIMPLEX, 1/2, (0, 0, 255), 1)

        except Exception as e:
            print("Something wrong: ", e)
    except:
        print("Something wrongs!")
        
    if show == True:
        show_img(img2, 'Detected Result(s)')
    if export == True:
        original_path = img_name.split('/')[1].split('.')[0]
        export_img(img2, original_path)
        
    # show_img(mask_r1, 'mask_r1')
    # show_img(mask_r2, 'mask_r2')
    # show_img(mask_r, 'mask_r')
    # show_img(target, 'target')
    # show_img(gblur, 'gblur')
    # show_img(edge_img, 'edge_img')

def get_class_name(classes, crop_name):
    labels = pd.read_csv('labels.csv')
    class_name = labels['Name'][classes[0]]
    export_class_name(class_name, crop_name)
    return class_name; 

def export_class_name(class_name, crop_img):
    with open('Results/results.txt', 'a') as file:
        file.write(crop_img + ',' + class_name)
        file.write("\n")
    
def export_crop_img(crop, img_name, specific_str):
    crop_name = 'Results/crop_images/' + img_name.split('/')[1].split('.')[0] + "_" + specific_str
    cv2.imwrite(crop_name + '.png', crop)
    return crop_name;
    
def export_img(img, img_name):
    name = 'Results/r_' + img_name
    cv2.imwrite(name + ".png", img)

def show_img(img, title):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size

    plt.axis('off')
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def show_img_gray(img, title):
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

def main():
    print("--- Run Main Function ---")
    img_name = 'Test/Test_00_06.jpg'
    img = read_img(img_name)
    detect_img(img, img_name, show=False, export=True)
    
main()