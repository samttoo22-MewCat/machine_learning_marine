import numpy as np
import os
import sys
import pandas as pd
from keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, Flatten
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import tensorflow as tf
from tensorflow import keras 
from sklearn.model_selection import train_test_split
from filter_dataset import filter_data

def process_data(ship_data: list, all_ship_data: list):
    # 使用所有類別對 Navigational_Status 進行OneHot編碼
    encoder = OneHotEncoder(categories=["all_navigational_status_categories"])
    encoded_status = encoder.fit_transform(ship_data[["Navigational_Status"]]).toarray()
    encoded_status_df = pd.DataFrame(encoded_status, columns=[f"status_{i}" for i in range(16)])

    encoder = OneHotEncoder(categories=["all_ship_and_cargo_type_categories"])
    encoded_status = encoder.fit_transform(ship_data[["Ship_and_Cargo_Type"]]).toarray()
    encoded_type_df = pd.DataFrame(encoded_status, columns=[f"type_{i}" for i in "all_ship_and_cargo_type_categories"])
    
    # 标准化数值特征
    scaler = MinMaxScaler()
    scaler.fit(all_ship_data[["SOG", "Longitude", "Latitude", "COG"]])
    scaled_num_features = scaler.fit_transform(ship_data[["SOG", "Longitude", "Latitude", "COG"]])
    scaled_num_features_df = pd.DataFrame(scaled_num_features, columns=["SOG", "Longitude", "Latitude", "COG"])
    
    # 合并特征
    processed_data = pd.concat([encoded_status_df, encoded_type_df, scaled_num_features_df], axis=1)
    
    return processed_data



def load_or_create_model(model_path):
    sequence_length = 100
    n_features = 22
    n_targets = 4
    embedding_dim = 32
    num_heads = 8
    ff_dim = 64
    dropout_rate = 0.1

    def create_transformer_model():
        # 编码器模块
        encoder_inputs = Input(shape=(sequence_length, n_features))
        enc_embeddings = tf.keras.layers.Dense(embedding_dim)(encoder_inputs)
        enc_embeddings = LayerNormalization()(enc_embeddings)
        enc_outputs = tf.keras.layers.Dropout(dropout_rate)(enc_embeddings)

        # 解码器模块
        decoder_inputs = Input(shape=(sequence_length, n_targets))
        dec_embeddings = tf.keras.layers.Dense(embedding_dim)(decoder_inputs)
        dec_embeddings = LayerNormalization()(dec_embeddings)
        dec_outputs = tf.keras.layers.Dropout(dropout_rate)(dec_embeddings)

        # 多头注意力层
        enc_attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(enc_outputs, enc_outputs)
        enc_attn_output = LayerNormalization()(enc_attn_output + enc_outputs)

        dec_attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(dec_outputs, enc_attn_output)
        dec_attn_output = LayerNormalization()(dec_attn_output + dec_outputs)

        # 前馈网络
        ff = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embedding_dim),
        ])

        enc_ff_output = ff(enc_attn_output)
        enc_ff_output = LayerNormalization()(enc_ff_output + enc_attn_output)
        enc_ff_output = tf.keras.layers.Dropout(dropout_rate)(enc_ff_output)

        dec_ff_output = ff(dec_attn_output)
        dec_ff_output = LayerNormalization()(dec_ff_output + dec_attn_output)
        dec_ff_output = tf.keras.layers.Dropout(dropout_rate)(dec_ff_output)

        # 连接编码器和解码器
        encoder = keras.Model(inputs=encoder_inputs, outputs=enc_ff_output)
        decoder = keras.Model(inputs=[decoder_inputs, enc_ff_output], outputs=dec_ff_output)

        # 构建Transformer模型
        transformer_outputs = Dense(n_targets)(decoder(decoder_inputs, encoder(encoder_inputs)))
        transformer = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=transformer_outputs)

        # 编译模型
        transformer.compile(optimizer="adam", loss="mse")

        return transformer

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print(f"已加载船舶现有模型")
        return model
    else:
        model = create_transformer_model()
        model.save(model_path)
        print(f"为船舶创建了新模型")
        return model