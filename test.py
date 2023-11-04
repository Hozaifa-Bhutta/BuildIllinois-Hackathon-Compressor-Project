import cv2

cv2.namedWindow("preview")
image = cv2.imread("test.jpg")

saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")

#split the image into its left and right halves and compute their saliency


topHalf = image[0: image.shape[0] // 2, 0: image.shape[1]]
(_, topS) = saliency.computeSaliency(topHalf)
topS = (topS * 255).astype("uint8")

bottomHalf = image[image.shape[0] // 2: image.shape[0], 0: image.shape[1]]
(_, bottomS) = saliency.computeSaliency(bottomHalf)
bottomS = (bottomS * 255).astype("uint8")
# rightHalf = cv2.resize(saliencyMap, (saliencyMap.shape[1] // 2, saliencyMap.shape[0]))

cv2.imshow("Image", image)
cv2.imshow("Saliency", saliencyMap)


# cv2.imshow("Both", cv2.vconcat([topHalf, bottomHalf]))
# cv2.imshow("Both Saliency", cv2.vconcat([topS, bottomS]))

# cv2.imshow("Difference", cv2.absdiff(cv2.vconcat([topS, bottomS]), saliencyMap))
cv2.waitKey(0)