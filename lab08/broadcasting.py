import tensorflow as tf

sess = tf.InteractiveSession()

# 사용시 주의할 것
# 같은 shape 내에서 연산을 권장

#http://excelsior-cjh.tistory.com/entry/Matrix-Broadcasting-%ED%96%89%EB%A0%AC%EC%9D%98-%EB%B8%8C%EB%A1%9C%EB%93%9C%EC%BA%90%EC%8A%A4%ED%8C%85
#Broadcast의 사전적인 의미는 '퍼뜨리다'라는 뜻이 있는데,
#이와 마찬가지로 두 행렬 A, B 중 크기가 작은 행렬을 크기가 큰 행렬과 모양(shape)이 맞게끔 늘려주는 것을 의미한다.

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])

print((matrix1 + matrix2).eval())



