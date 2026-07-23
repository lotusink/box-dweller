---
author: []
date:
  create: '20260723'
  modified: '20260724'
note: ''
---
# 盒中人 (Box Dweller)

![status](https://img.shields.io/badge/status-prototype-e8a33d?style=flat-square)
[![license](https://img.shields.io/badge/license-MIT-4a7c9e?style=flat-square)](LICENSE)
![react](https://img.shields.io/badge/react-19-4a7c9e?style=flat-square&logo=react&logoColor=white)
![vite](https://img.shields.io/badge/vite-8-646CFF?style=flat-square&logo=vite&logoColor=white)
![three.js](https://img.shields.io/badge/three.js-000000?style=flat-square&logo=three.js&logoColor=white)

## 概述

盒中人探索大语言模型与 3D 引擎之间的交互协议——模型如何在一个持续演化的三维空间里表达意图,引擎如何确定性地执行。为此,我们构建了一个体素化的小世界作为验证环境,在其上定义分层原语作为模型的动作接口,让模型通过对话逐步完成空间的构造。

我们希望模型的能力落在世界本身:握有建造权限,以离散原语的方式参与三维空间的构造,成为世界的共同作者。

## 项目结构

```
box-dweller/
├── docs/
│   ├── design.md          系统设计
│   ├── adr/               决策日志
│   └── research/          调研笔记
├── prototype/             验证环境(体素世界)
└── media/                 截图与录屏
```

## 路线图

- [x] 创建项目原型
- [ ] 优化代码结构
- [ ] 优化 L0 原语协议
- [ ] 优化大模型反思