import step01
import step02clean
import step03complete
import step04smooth
import step05to_sample
import step05to_sample_mttc
import step06sampling
import step06samplinge
import step06change
import step07pca
import visualizeXY
import gc
from step07pca import save_dir
#save_dir = r"E:\0little\read\CQSkyEyedata5\location5t"

def main_pipeline():
    raw_pkl_path = r"E:\0little\read\CQSkyEyedata5\location5\5_trajectory.pkl"
    print("=== 开始运行集成流水线 ===")

    df_east, df_west = step01.split_traffic_data(raw_pkl_path)
    df_east.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_flows_east.csv", index=False,
                          encoding='utf-8-sig')
    print(f"01主控脚本已接收到数据，上下流数据已分")

    df_east_cleaned, df_west_cleaned = step02clean.data_clean(df_east,df_west)
    print(f"02数据初步清理（长宽异常值、速度加速度异常值以及单点漂移插值的删除与修复）")
    # df_east_cleaned.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_flows_east_cleaned.csv"
    #                       , index=False,encoding='utf-8-sig')

    df_east_comp, df_west_comp = step03complete.data_complete(
        df_east_cleaned, df_west_cleaned, df_east, df_west )
    # df_east_comp.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_flows_east_comp.csv", index=False,
    #                       encoding='utf-8-sig')
    print('step3:数据的时序补全已完成')
    print('补全后的东西向数据分别为df_east_comp,df_west_comp')

    df_east_smooth, df_west_smooth = step04smooth.data_smooth(df_east_comp,df_west_comp)
    # df_east_smooth.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_flows_east_smooth.csv", index=False,
    #                       encoding='utf-8-sig')
    print('step4:速度、加速度及相关距离已完成平滑')

    df_east_sample, df_west_sample = step05to_sample_mttc.data_features(df_east_smooth, df_west_smooth)
    print(f"样本化数据已完成")
    print(f"生成的交互特征列: {[c for c in df_east_sample.columns if 'Dist' in c]}")
    print(f"核算的动态安全指标: TTC={df_east_sample['TTC'].mean():.2f}, THW={df_east_sample['Time_Headway'].mean():.2f}")
    print('step5将周围车的ID转化为欧氏距离已完成，核算TTC和mTTC')
    df_east_sample.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_flows_east_sample1.csv"
                          , index=False, encoding='utf-8-sig')
    df_west_sample.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_flows_west_sample1.csv"
                          , index=False, encoding='utf-8-sig')

    print('step6：对东西向数据进行样本截取，输出df_west_sampling,df_east_sampling')
    #读取df_east_sample,df_west_sample以及未计算欧氏距离之前，尚存邻车ID的数据，先输出东西方向的左右变道数据，之后输出东西向的全样本
    df_left_change, df_right_change = step06change.extract_lane_change_samples(
                                    df_east_sample,df_west_sample,df_east_smooth,df_west_smooth)
    df_left_change.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_left_change.csv", index=False,
                         encoding='utf-8-sig')
    df_right_change.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_right_change.csv", index=False,
                    encoding='utf-8-sig')
    #对东西向数据进行样本截取，包括左右变道和跟驰，根据方向，输出df_west_sampling,df_east_sampling
    df_west_sampling = step06sampling.process_west(df_west_sample)
    df_east_sampling = step06samplinge.process_east(df_east_sample)
    visualizeXY.visualize_trajectory_samples(df_east_smooth, df_west_smooth,
                                 df_east_sampling, df_west_sampling,save_dir=save_dir)
    #df_east_sampling.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_flows_east_sampling.csv"
    #                      , index=False,encoding='utf-8-sig')

    print('step7：对十六个指标进行归一化与降维，并进行样本分割，按照8:2分割训练集、验证集，输出储存df_train,df_val')
    df_train, df_val = step07pca.data_pca_divide(df_east_sampling,df_west_sampling)
    df_train.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_flow_train.csv", index=False,
                         encoding='utf-8-sig')
    df_val.to_csv(r"E:\0little\read\CQSkyEyedata5\location5t\traffic_flow_val.csv", index=False,
                    encoding='utf-8-sig')
    print(f"数据清理、补全、平滑、归一、降维与样本分割已完成，\n样本与训练集已储存{save_dir}")
if __name__ == "__main__":
    main_pipeline()
