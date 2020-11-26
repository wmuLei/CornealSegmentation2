# a edge-aware deep convolutional network (EA-Net) for segmenting corneal micro-layers depicted on OCT images


def EE-Net(shape, classes=1):
    inputs = Input(shape) # [256, 256, 1]
    pool1 = BatchNormalization()(inputs)
    
    global conv1a, conv2a, conv3a, conv4a, conv5a
    conv1a=None
    conv2a=None
    conv3a=None
    conv4a=None
    conv5a=None

    if conv1a is not None: pool1 = merge([pool1, conv1a], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 32, (3, 3));    conv1 = CONV2D(conv0, 32, (3, 3));     edge1 = Subtract()([conv0, conv1]) 
    conv1 = merge([conv1, edge1], mode='concat', concat_axis=3);    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1);  # 256/2

    if conv2a is not None: pool1 = merge([pool1, conv2a], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 64, (3, 3));    conv2 = CONV2D(conv0, 64, (3, 3));     edge1 = Subtract()([conv0, conv2]);
    conv2 = merge([conv2, edge1], mode='concat', concat_axis=3);    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2);  # 256/4

    if conv3a is not None: pool1 = merge([pool1, conv3a], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 128, (3, 3));    conv3 = CONV2D(conv0, 128, (3, 3));     edge1 = Subtract()([conv0, conv3]);
    conv3 = merge([conv3, edge1], mode='concat', concat_axis=3);    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv3);  # 256/8

    if conv4a is not None: pool1 = merge([pool1, conv4a], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 256, (3, 3));    conv4 = CONV2D(conv0, 256, (3, 3));     edge1 = Subtract()([conv0, conv4]);
    conv4 = merge([conv4, edge1], mode='concat', concat_axis=3);    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv4);  # 256/16

    if conv5a is not None: pool1 = merge([pool1, conv5a], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 512, (3, 3));    conv5 = CONV2D(conv0, 512, (3, 3));     edge1 = Subtract()([conv0, conv5]);
    conv5 = merge([conv5, edge1], mode='concat', concat_axis=3);    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv5);  # 256/32

    #----------------------------------------------
    conv0 = CONV2D(pool1, 1024, (3, 3));    conv6 = CONV2D(conv0, 1024, (3, 3));     edge1 = Subtract()([conv0, conv6]);
    conv6 = merge([conv6, edge1], mode='concat', concat_axis=3);  # 256/32
    #----------------------------------------------

    merg1 = UpSampling2D(size=(2, 2))(conv6);
    merg1 = merge([merg1, conv5], mode='concat', concat_axis=3) # 256/16
    conv0 = CONV2D(merg1, 512, (3, 3));    conv5 = CONV2D(conv0, 512, (3, 3));     edge1 = Subtract()([conv0, conv5]);
    conv5a = merge([conv5, edge1], mode='concat', concat_axis=3);
    
    merg1 = UpSampling2D(size=(2, 2))(conv5a);
    merg1 = merge([merg1, conv4], mode='concat', concat_axis=3) # 256/8
    conv0 = CONV2D(merg1, 256, (3, 3));    conv4 = CONV2D(conv0, 256, (3, 3));     edge1 = Subtract()([conv0, conv4]);
    conv4a = merge([conv4, edge1], mode='concat', concat_axis=3); 

    merg1 = UpSampling2D(size=(2, 2))(conv4a);
    merg1 = merge([merg1, conv3], mode='concat', concat_axis=3) # 256/4
    conv0 = CONV2D(merg1, 128, (3, 3));    conv3 = CONV2D(conv0, 128, (3, 3));     edge1 = Subtract()([conv0, conv3]);
    conv3a = merge([conv3, edge1], mode='concat', concat_axis=3); 

    merg1 = UpSampling2D(size=(2, 2))(conv3a);
    merg1 = merge([merg1, conv2], mode='concat', concat_axis=3) # 256/4
    conv0 = CONV2D(merg1, 64, (3, 3));    conv2 = CONV2D(conv0, 64, (3, 3));     edge1 = Subtract()([conv0, conv2]);
    conv2a = merge([conv2, edge1], mode='concat', concat_axis=3);

    merg1 = UpSampling2D(size=(2, 2))(conv2a);
    merg1 = merge([merg1, conv1], mode='concat', concat_axis=3) # 256/2
    conv0 = CONV2D(merg1, 32, (3, 3));    conv1 = CONV2D(conv0, 32, (3, 3));     edge1 = Subtract()([conv0, conv1]);
    conv1a = merge([conv1, edge1], mode='concat', concat_axis=3);

    conv0 = CONV2D(conv1a, classes, (1, 1), activation='sigmoid')
    model = Model(input=inputs, output=conv0)
    model.summary() 
    return model
