# a edge-aware deep convolutional network (EA-Net) for segmenting corneal micro-layers depicted on OCT images


def EA_Net(shape, classes=1):
    inputs = Input(shape) 
    conv0 = BatchNormalization()(inputs)

    conv0 = CONV2D(conv0, 32, (3, 3))
    conv1 = CONV2D(conv0, 32, (3, 3)); edge1 = Subtract()([conv0, conv1]); edge1 = Add()([edge1, Lambda(lambda x: K.abs(x))(edge1)]);
    conv1 = Add()([conv1, edge1]);    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1);  

    conv0 = CONV2D(pool1, 64, (3, 3))
    conv2 = CONV2D(conv0, 64, (3, 3)); edge2 = Subtract()([conv0, conv2]); edge2 = Add()([edge2, Lambda(lambda x: K.abs(x))(edge2)]);
    conv2 = Add()([conv2, edge2]);    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2);  

    conv0 = CONV2D(pool2, 128, (3, 3))
    conv3 = CONV2D(conv0, 128, (3, 3)); edge3 = Subtract()([conv0, conv3]); edge3 = Add()([edge3, Lambda(lambda x: K.abs(x))(edge3)]);
    conv3 = Add()([conv3, edge3]);    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3);  

    conv0 = CONV2D(pool3, 256, (3, 3))
    conv4 = CONV2D(conv0, 256, (3, 3)); edge4 = Subtract()([conv0, conv4]); edge4 = Add()([edge4, Lambda(lambda x: K.abs(x))(edge4)]);
    conv4 = Add()([conv4, edge4]);    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4);  

    conv0 = CONV2D(pool4, 512, (3, 3))
    conv5 = CONV2D(conv0, 512, (3, 3)); edge5 = Subtract()([conv0, conv5]); edge5 = Add()([edge5, Lambda(lambda x: K.abs(x))(edge5)]);
    conv5 = Add()([conv5, edge5]);    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5);  

    #----------------------------------------------
    conv0 = CONV2D(pool5, 1024, (3, 3))
    conv6 = CONV2D(conv0, 1024, (3, 3)); edge6 = Subtract()([conv0, conv6]); edge6 = Add()([edge6, Lambda(lambda x: K.abs(x))(edge6)]);
    conv6 = Add()([conv6, edge6]);  
    #----------------------------------------------

    up1 = UpSampling2D(size=(2, 2))(conv6);
    merg1 = merge([up1, conv5], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 512, (3, 3))
    conv7 = CONV2D(conv0, 512, (3, 3)); edge7 = Subtract()([conv0, conv7]); edge7 = Add()([edge7, Lambda(lambda x: K.abs(x))(edge7)]);
    conv7 = Add()([conv7, edge7]);
    
    up1 = UpSampling2D(size=(2, 2))(conv7);
    merg1 = merge([up1, conv4], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 256, (3, 3))
    conv8 = CONV2D(conv0, 256, (3, 3)); edge8 = Subtract()([conv0, conv8]); edge8 = Add()([edge8, Lambda(lambda x: K.abs(x))(edge8)]);
    conv8 = Add()([conv8, edge8]);

    up1 = UpSampling2D(size=(2, 2))(conv8);
    merg1 = merge([up1, conv3], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 128, (3, 3))
    conv9 = CONV2D(conv0, 128, (3, 3)); edge9 = Subtract()([conv0, conv9]); edge9 = Add()([edge9, Lambda(lambda x: K.abs(x))(edge9)]);
    conv9 = Add()([conv9, edge9]);

    up1 = UpSampling2D(size=(2, 2))(conv9);
    merg1 = merge([up1, conv2], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 64, (3, 3))
    convA = CONV2D(conv0, 64, (3, 3)); edgeA = Subtract()([conv0, convA]); edgeA = Add()([edgeA, Lambda(lambda x: K.abs(x))(edgeA)]);
    convA = Add()([convA, edgeA]);
    
    up1 = UpSampling2D(size=(2, 2))(convA);
    merg1 = merge([up1, conv1], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 32, (3, 3))
    convB = CONV2D(conv0, 32, (3, 3)); edgeB = Subtract()([conv0, convB]); edgeB = Add()([edgeB, Lambda(lambda x: K.abs(x))(edgeB)]);
    convB = Add()([convB, edgeB]);

    convB = CONV2D(convB, classes, (1, 1), activation='sigmoid')
    model = Model(input=inputs, output=convB)
    model.summary() 
    return model
