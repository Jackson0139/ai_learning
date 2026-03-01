import pandas as pd
import json
from datetime import datetime


class ReportGenerator:
    """
    报告生成器
    生成模型监控报告
    """
    
    def __init__(self, output_dir='reports'):
        self.output_dir = output_dir
    
    def generate_full_report(self, model_discrimination, model_stability, 
                            model_ranking, feature_stability, output_file=None):
        """
        生成完整报告
        
        参数:
            model_discrimination: 模型区分度监控结果
            model_stability: 模型稳定性监控结果
            model_ranking: 模型排序性分析结果
            feature_stability: 特征稳定性监控结果
            output_file: 输出文件路径
        
        返回:
            报告字典
        """
        report = {
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_summary': {
                'discrimination': model_discrimination.get_summary() if model_discrimination else None,
                'stability': model_stability.get_summary() if model_stability else None,
            },
            'model_ranking': model_ranking.get_summary() if model_ranking else None,
            'feature_stability': feature_stability.get_summary() if feature_stability else None
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def export_to_excel(self, model_discrimination, model_stability, 
                       model_ranking, feature_stability, output_file):
        """
        导出到Excel（多个sheet）
        """
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: 模型KS
            if model_discrimination and model_discrimination.ks_results is not None:
                model_discrimination.ks_results.to_excel(
                    writer, sheet_name='模型KS', index=False
                )
            
            # Sheet 2: 模型PSI
            if model_stability and model_stability.psi_results is not None:
                model_stability.psi_results.to_excel(
                    writer, sheet_name='模型PSI', index=False
                )
            
            # Sheet 3: 模型排序性
            if model_ranking and model_ranking.ranking_results is not None:
                model_ranking.ranking_results.to_excel(
                    writer, sheet_name='模型排序性', index=False
                )
            
            # Sheet 4+: 特征PSI和排序性
            if feature_stability:
                # 汇总所有特征的PSI
                all_psi = []
                for feature, psi_df in feature_stability.psi_results.items():
                    psi_df_copy = psi_df.copy()
                    psi_df_copy['feature'] = feature
                    all_psi.append(psi_df_copy)
                
                if all_psi:
                    pd.concat(all_psi, ignore_index=True).to_excel(
                        writer, sheet_name='特征PSI汇总', index=False
                    )
                
                # 每个特征的排序性
                for feature, binning_df in feature_stability.binning_results.items():
                    sheet_name = f'{feature}_排序性'[:31]  # Excel sheet名最多31字符
                    binning_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 最后一个sheet: 模型分与特征分箱策略
            binning_info = []
            
            # 添加模型分数分箱信息
            if model_ranking and hasattr(model_ranking, 'data_distribution') and model_ranking.data_distribution:
                dist = model_ranking.data_distribution
                strategy = model_ranking.binner.get_actual_strategy()
                binning_info.append({
                    '指标': '模型分数',
                    '分箱策略': strategy,
                    '数据分布': '均匀' if dist['is_uniform'] else '不均匀',
                    '唯一值数量': dist['unique_count'],
                    '样本数量': dist['sample_count'],
                    '最小值': dist['min'],
                    '最大值': dist['max'],
                    '平均值': dist['mean'],
                    '标准差': dist['std'],
                    '分箱边界': ', '.join([f'{float(x):.2f}' for x in model_ranking.binner.get_bin_edges()])
                })
            
            # 添加特征分箱信息
            if feature_stability:
                for feature in feature_stability.feature_cols:
                    dist = feature_stability.data_distribution[feature]
                    strategy = feature_stability.binners[feature].get_actual_strategy()
                    binning_info.append({
                        '指标': feature,
                        '分箱策略': strategy,
                        '数据分布': '均匀' if dist['is_uniform'] else '不均匀',
                        '唯一值数量': dist['unique_count'],
                        '样本数量': dist['sample_count'],
                        '最小值': dist['min'],
                        '最大值': dist['max'],
                        '平均值': dist['mean'],
                        '标准差': dist['std'],
                        '分箱边界': ', '.join([f'{float(x):.2f}' for x in feature_stability.binners[feature].get_bin_edges()])
                    })
            
            if binning_info:
                pd.DataFrame(binning_info).to_excel(
                    writer, sheet_name='模型分与特征分箱策略', index=False
                )
        
        print(f"报告已导出到: {output_file}")
    
    def print_summary(self, report):
        """
        打印报告摘要到控制台
        """
        print("=" * 60)
        print("模型监控报告")
        print("=" * 60)
        print(f"报告时间: {report['report_time']}")
        print()
        
        # 模型区分度
        if report['model_summary']['discrimination']:
            print("【模型区分度 (KS)】")
            disc = report['model_summary']['discrimination']
            for item in disc['weekly_ks']:
                print(f"  第{item['week']}周: KS={item['ks']}, 样本数={item['sample_count']}")
            print(f"  平均KS: {disc['avg_ks']}")
            print()
        
        # 模型稳定性
        if report['model_summary']['stability']:
            print("【模型稳定性 (PSI)】")
            stab = report['model_summary']['stability']
            print(f"  基准周: 第{stab['base_week']}周")
            for item in stab['weekly_psi']:
                base_mark = " (基准)" if item['is_base'] else ""
                print(f"  第{item['week']}周: PSI={item['psi']}{base_mark}")
            print(f"  平均PSI: {stab['avg_psi']}")
            print()
        
        # 特征稳定性
        if report['feature_stability']:
            print("【特征稳定性】")
            for feature, feat_data in report['feature_stability']['features'].items():
                print(f"  {feature}: 平均PSI={feat_data['avg_psi']}, 最大PSI={feat_data['max_psi']}")
            print()
        
        print("=" * 60)
