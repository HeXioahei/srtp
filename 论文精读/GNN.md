图（graph）有四个信息，顶点（v）、边（e）、整个图（u）、连接性。前三者的每个都可以由一个向量来表示。而连接性可以由邻接矩阵来表示。

每个GNN层都有分别属于v、e、u的三个全连接层（MLP），每个全连接层的输出维度等于输入维度等于向量的长度。经过MLP后，图的结构没有发生变化，连接性没有发生变化，我们可以提取v、e、u的信息，但是三个独立的MLP利用不到图的“连接性”。

v、e、u三者之间可以通过池化层汇聚（pooling）来相互预测。也就可以用同样的思路来利用连接性。

![131e7d0c-de6d-45c7-8b3d-0dc243d8bd41](file:///C:/Users/Lenovo/Pictures/Typedown/131e7d0c-de6d-45c7-8b3d-0dc243d8bd41.png)

![bfe002e9-9d02-49db-8b95-1c360297d09a](file:///C:/Users/Lenovo/Pictures/Typedown/bfe002e9-9d02-49db-8b95-1c360297d09a.png)

![41c18570-bdfe-4a3b-81f7-97e516a1743a](file:///C:/Users/Lenovo/Pictures/Typedown/41c18570-bdfe-4a3b-81f7-97e516a1743a.png)

![5ceb6329-ad57-4427-a467-73abfca53016](file:///C:/Users/Lenovo/Pictures/Typedown/5ceb6329-ad57-4427-a467-73abfca53016.png)

![](file:///C:/Users/Lenovo/Pictures/Screenshots/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-17%20174619.png)

![c8ed2cc0-c212-4605-a669-9c6831903815](file:///C:/Users/Lenovo/Pictures/Typedown/c8ed2cc0-c212-4605-a669-9c6831903815.png)


