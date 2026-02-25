#!/usr/bin/env python3
"""
Code Review - 代码审查
对自己的代码进行系统性审查和改进建议
"""

import ast
import inspect
from pathlib import Path
from typing import List, Dict


class CodeReviewer:
    """
    代码审查器
    
    审查维度:
    1. 代码风格 (PEP8)
    2. 文档完整性
    3. 错误处理
    4. 性能考虑
    5. 可测试性
    6. 可维护性
    """
    
    def __init__(self):
        self.issues = []
        self.suggestions = []
    
    def review_file(self, filepath: str) -> Dict:
        """审查单个文件"""
        print(f"\n🔍 审查: {filepath}")
        print("-" * 60)
        
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        self.issues = []
        self.suggestions = []
        
        # 1. 检查文档字符串
        self._check_docstrings(content)
        
        # 2. 检查类型注解
        self._check_type_hints(content)
        
        # 3. 检查错误处理
        self._check_error_handling(content)
        
        # 4. 检查代码长度
        self._check_code_length(lines)
        
        # 5. 检查注释
        self._check_comments(lines)
        
        # 6. 检查硬编码
        self._check_magic_numbers(content)
        
        report = {
            'file': filepath,
            'total_lines': len(lines),
            'issues': self.issues,
            'suggestions': self.suggestions,
            'score': self._calculate_score()
        }
        
        self._print_report(report)
        return report
    
    def _check_docstrings(self, content: str):
        """检查文档字符串"""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    name = node.name
                    if not ast.get_docstring(node):
                        self.issues.append({
                            'type': 'missing_docstring',
                            'target': name,
                            'severity': 'medium',
                            'message': f"{node.__class__.__name__} '{name}' 缺少文档字符串"
                        })
        except:
            pass
    
    def _check_type_hints(self, content: str):
        """检查类型注解"""
        # 简单检查是否使用了typing
        if 'from typing import' not in content and 'import typing' not in content:
            self.suggestions.append({
                'type': 'type_hints',
                'message': '考虑添加类型注解提高代码可读性',
                'example': 'def func(x: float) -> float:'
            })
    
    def _check_error_handling(self, content: str):
        """检查错误处理"""
        try:
            tree = ast.parse(content)
            
            has_try_except = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    has_try_except = True
                    break
            
            if not has_try_except:
                self.suggestions.append({
                    'type': 'error_handling',
                    'message': '建议添加try-except处理潜在异常'
                })
        except:
            pass
    
    def _check_code_length(self, lines: List[str]):
        """检查代码长度"""
        long_lines = []
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                long_lines.append(i)
        
        if long_lines:
            self.issues.append({
                'type': 'long_lines',
                'lines': long_lines[:5],
                'severity': 'low',
                'message': f'发现 {len(long_lines)} 行超过100字符'
            })
    
    def _check_comments(self, lines: List[str]):
        """检查注释"""
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        
        ratio = comment_lines / (code_lines + 1)
        if ratio < 0.1:
            self.suggestions.append({
                'type': 'comments',
                'message': f'注释比例较低 ({ratio*100:.1f}%)，建议增加解释性注释'
            })
    
    def _check_magic_numbers(self, content: str):
        """检查魔术数字"""
        # 简单检查
        magic_patterns = ['0.5', '100', '0.25', '3.45']
        found = []
        for pattern in magic_patterns:
            if pattern in content and f'{pattern}  # ' not in content:
                found.append(pattern)
        
        if found:
            self.suggestions.append({
                'type': 'magic_numbers',
                'message': f'发现魔术数字 {found}，建议定义为常量',
                'example': 'TARGET_SPLITTING = 0.25  # 25%'
            })
    
    def _calculate_score(self) -> int:
        """计算代码质量分数"""
        score = 100
        
        for issue in self.issues:
            if issue['severity'] == 'high':
                score -= 10
            elif issue['severity'] == 'medium':
                score -= 5
            else:
                score -= 2
        
        score -= len(self.suggestions) * 1
        
        return max(0, score)
    
    def _print_report(self, report: Dict):
        """打印审查报告"""
        print(f"代码行数: {report['total_lines']}")
        print(f"质量评分: {report['score']}/100")
        print()
        
        if report['issues']:
            print("🚨 发现的问题:")
            for issue in report['issues']:
                icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(
                    issue.get('severity', 'low'), '⚪'
                )
                print(f"  {icon} {issue['message']}")
            print()
        
        if report['suggestions']:
            print("💡 改进建议:")
            for sug in report['suggestions']:
                print(f"  • {sug['message']}")
                if 'example' in sug:
                    print(f"    示例: {sug['example']}")
            print()
        
        if not report['issues'] and not report['suggestions']:
            print("✅ 代码质量良好！")


def review_all_files():
    """审查所有代码文件"""
    print("=" * 70)
    print("SRTP代码审查报告")
    print("=" * 70)
    
    reviewer = CodeReviewer()
    
    files = [
        'optimizer.py',
        'manufacturing.py', 
        'objective.py',
        'code_review.py'
    ]
    
    reports = []
    for f in files:
        if Path(f).exists():
            report = reviewer.review_file(f)
            reports.append(report)
    
    # 汇总
    print("\n" + "=" * 70)
    print("审查汇总")
    print("=" * 70)
    
    total_issues = sum(len(r['issues']) for r in reports)
    total_suggestions = sum(len(r['suggestions']) for r in reports)
    avg_score = np.mean([r['score'] for r in reports])
    
    print(f"审查文件数: {len(reports)}")
    print(f"总问题数: {total_issues}")
    print(f"总建议数: {total_suggestions}")
    print(f"平均评分: {avg_score:.1f}/100")
    
    if avg_score >= 90:
        print("\n✅ 整体代码质量优秀")
    elif avg_score >= 70:
        print("\n⚠️  整体代码质量良好，有改进空间")
    else:
        print("\n🔴 整体代码需要改进")
    
    return reports


if __name__ == "__main__":
    import numpy as np
    
    reports = review_all_files()
    
    # 保存报告
    with open('code_review_report.md', 'w') as f:
        f.write("# 代码审查报告\n\n")
        for r in reports:
            f.write(f"## {r['file']}\n")
            f.write(f"- 评分: {r['score']}/100\n")
            f.write(f"- 问题: {len(r['issues'])}\n")
            f.write(f"- 建议: {len(r['suggestions'])}\n\n")
    
    print("\n报告已保存: code_review_report.md")
