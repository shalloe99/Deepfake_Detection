import cv2, numpy as np, os, radialProfile, glob, pickle
from scipy.interpolate import griddata
from matplotlib import pyplot as plt




def loadData(num_iter, real_folder, fake_folder, out_pickle, N=300, epsilon=100):
    data = {}

    #fake data
    count = 0
    psd1D_total = np.zeros([num_iter, N])
    labels = np.zeros([num_iter])
    for subdir, dirs, files in os.walk(fake_folder):
        for file in files:        

            filename = os.path.join(subdir, file)
            
            img = cv2.imread(filename,0)
            
            # we crop the center
            h = int(img.shape[0]/3)
            w = int(img.shape[1]/3)
            img = img[h:-h,w:-w]

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)

            magnitude_spectrum = 20*np.log(np.abs(fshift))
            psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

            # Calculate the azimuthally averaged 1D power spectrum
            points = np.linspace(0,N,num=psd1D.size) # coordinates of a
            xi = np.linspace(0,N,num=N) # coordinates for interpolation

            interpolated = griddata(points,psd1D,xi,method='cubic')
            interpolated /= interpolated[0]

            psd1D_total[count,:] = interpolated             
            labels[count] = 0
            count+=1

            if count == num_iter:
                break
        if count == num_iter:
            break
                

    # real data
    count = 0
    psd1D_total2 = np.zeros([num_iter, N])
    labels2 = np.zeros([num_iter])
    for subdir, dirs, files in os.walk(real_folder):
        for file in files:        
            filename = os.path.join(subdir, file)
    
            img = cv2.imread(filename,0)
        
            # we crop the center
            h = int(img.shape[0]/3)
            w = int(img.shape[1]/3)
            img = img[h:-h,w:-w]

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            fshift += epsilon


            magnitude_spectrum = 20*np.log(np.abs(fshift))

            # Calculate the azimuthally averaged 1D power spectrum
            psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

            points = np.linspace(0,N,num=psd1D.size) # coordinates of a
            xi = np.linspace(0,N,num=N) # coordinates for interpolation

            interpolated = griddata(points,psd1D,xi,method='cubic')
            interpolated /= interpolated[0]

            psd1D_total2[count,:] = interpolated             
            labels2[count] = 1
            count+=1

            if count == num_iter:
                break
        if count == num_iter:
            break    
        

    psd1D_total_final = np.concatenate((psd1D_total,psd1D_total2), axis=0)
    labels_final = np.concatenate((labels,labels2), axis=0)

    data["data"] = psd1D_total_final
    data["label"] = labels_final

    output = open(out_pickle, 'wb')
    pickle.dump(data, output)
    output.close()

def multiclass_data(num_iter, filess, out_pickle, N=300, epsilon=100):
    data = {}

    #fake data - Deepfake
    psd1D_all = []
    labels_all = []

    count = 0
    psd1D_total = np.zeros([num_iter, N])
    labels = np.zeros([num_iter])
    for subdir, dirs, files in os.walk(filess[0]):
        for file in files:        

            filename = os.path.join(subdir, file)
            img = cv2.imread(filename,0)
            
            # we crop the center
            h = int(img.shape[0]/3)
            w = int(img.shape[1]/3)
            img = img[h:-h,w:-w]

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            fshift += 0.25*epsilon

            magnitude_spectrum = 20*np.log(np.abs(fshift))
            psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

            # Calculate the azimuthally averaged 1D power spectrum
            points = np.linspace(0,N,num=psd1D.size) # coordinates of a
            xi = np.linspace(0,N,num=N) # coordinates for interpolation

            interpolated = griddata(points,psd1D,xi,method='cubic')
            interpolated /= interpolated[0]

            psd1D_total[count,:] = interpolated             
            labels[count] = 0   # labels data i
            count+=1

            if count == num_iter:
                break
        if count == num_iter:
            break
    
    psd1D_all.append(psd1D_total)
    labels_all.append(labels)
        
    count = 0
    psd2D_total = np.zeros([num_iter, N])
    labels2 = np.zeros([num_iter])

    for subdir, dirs, files in os.walk(filess[1]):
        for file in files:        

            filename = os.path.join(subdir, file)
            img = cv2.imread(filename,0)
            
            # we crop the center
            h = int(img.shape[0]/3)
            w = int(img.shape[1]/3)
            img = img[h:-h,w:-w]

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            fshift += 1*epsilon

            magnitude_spectrum = 20*np.log(np.abs(fshift))
            psd2D = radialProfile.azimuthalAverage(magnitude_spectrum)

            # Calculate the azimuthally averaged 1D power spectrum
            points = np.linspace(0,N,num=psd2D.size) # coordinates of a
            xi = np.linspace(0,N,num=N) # coordinates for interpolation

            interpolated = griddata(points,psd2D,xi,method='cubic')
            interpolated /= interpolated[0]

            psd2D_total[count,:] = interpolated             
            labels2[count] = 1   # labels data i
            count+=1

            if count == num_iter:
                break
        if count == num_iter:
            break
    psd1D_all.append(psd2D_total)
    labels_all.append(labels2)

    count = 0
    psd3D_total = np.zeros([num_iter, N])
    labels3 = np.zeros([num_iter])
    for subdir, dirs, files in os.walk(filess[2]):
        for file in files:        

            filename = os.path.join(subdir, file)
            img = cv2.imread(filename,0)
            
            # we crop the center
            h = int(img.shape[0]/3)
            w = int(img.shape[1]/3)
            img = img[h:-h,w:-w]

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            fshift += 2*epsilon

            magnitude_spectrum = 20*np.log(np.abs(fshift))
            psd3D = radialProfile.azimuthalAverage(magnitude_spectrum)

            # Calculate the azimuthally averaged 1D power spectrum
            points = np.linspace(0,N,num=psd3D.size) # coordinates of a
            xi = np.linspace(0,N,num=N) # coordinates for interpolation

            interpolated = griddata(points,psd3D,xi,method='cubic')
            interpolated /= interpolated[0]

            psd3D_total[count,:] = interpolated             
            labels3[count] = 2   # labels data i
            count+=1

            if count == num_iter:
                break
        if count == num_iter:
            break
    psd1D_all.append(psd3D_total)
    labels_all.append(labels3)
        

    psd1D_total_final = np.concatenate(psd1D_all, axis=0)
    labels_final = np.concatenate(labels_all, axis=0)

    data["data"] = psd1D_total_final
    data["label"] = labels_final

    output = open(out_pickle, 'wb')
    pickle.dump(data, output)
    output.close()


def ezdata(num_iter, filess, out_pickle, N=300, epsilon=100):
    data = {}

    #fake data - Deepfake
    psd1D_all = []
    labels_all = []


    for i in range(len(filess)):
        count = 0
        psd1D_total = np.zeros([num_iter, N])
        labels = np.zeros([num_iter])

        for subdir, dirs, files in os.walk(filess[i]):
            for file in files:        

                filename = os.path.join(subdir, file)
                img = cv2.imread(filename,0)
                
                # we crop the center
                h = int(img.shape[0]/3)
                w = int(img.shape[1]/3)
                img = img[h:-h,w:-w]

                f = np.fft.fft2(img)
                fshift = np.fft.fftshift(f)
                fshift += 0.25*epsilon

                magnitude_spectrum = 20*np.log(np.abs(fshift))
                psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

                # Calculate the azimuthally averaged 1D power spectrum
                points = np.linspace(0,N,num=psd1D.size) # coordinates of a
                xi = np.linspace(0,N,num=N) # coordinates for interpolation

                interpolated = griddata(points,psd1D,xi,method='cubic')
                interpolated /= interpolated[0]

                psd1D_total[count,:] = interpolated             
                labels[count] = i   # labels data i
                count+=1

                if count == num_iter:
                    break
            if count == num_iter:
                break
        
        psd1D_all.append(psd1D_total)
        labels_all.append(labels)       

    psd1D_total_final = np.concatenate(psd1D_all, axis=0)
    labels_final = np.concatenate(labels_all, axis=0)

    data["data"] = psd1D_total_final
    data["label"] = labels_final

    output = open(out_pickle, 'wb')
    pickle.dump(data, output)
    output.close()