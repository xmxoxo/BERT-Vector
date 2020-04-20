#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'


import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='BERTVector',
    version='0.3.6',
    description='extract vector from BERT pre-train model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='BERT vector extraction tool',
    install_requires=[],
    packages=setuptools.find_packages(),
    author='He xi',
    author_email='xmhexi@qq.com',
    url='https://github.com/xmxoxo/BERT-Vector',
    classifiers=[
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable'
        'Development Status :: 5 - Production/Stable',  # 当前开发进度等级（测试版，正式版等）
        'Intended Audience :: Developers',  # 模块适用人群
        'Topic :: Software Development :: Code Generators',  # 给模块加话题标签
        'License :: OSI Approved :: MIT License',  # 模块的license

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    project_urls={  # 项目相关的额外链接
        'Blog': 'https://blog.csdn.net/xmxoxo',
    },
    entry_points={
        'console_scripts': [
            'BERTVector = BERTVector.BERTVector:main_cli', #注意 mypackage是命令名称，=后面的是包名以及函数名
            ]
        }
)



