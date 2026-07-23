# 盒中人原型 (Box Dweller Prototype)

![status](https://img.shields.io/badge/status-prototype-e8a33d?style=flat-square)
![react](https://img.shields.io/badge/react-19-4a7c9e?style=flat-square&logo=react&logoColor=white)
![vite](https://img.shields.io/badge/vite-8-646CFF?style=flat-square&logo=vite&logoColor=white)
![three.js](https://img.shields.io/badge/three.js-000000?style=flat-square&logo=three.js&logoColor=white)

## 概述

本目录是盒中人的原型体,一个体素小世界。玩家以火柴人形态在场景中走动,通过中文对话向站在世界中央的"梦中神"下达指令,引擎把模型输出的 L0 / L1 原语确定性地写入几何、材质、光源与体素,让世界随着对话生长。

## 配置

复制 `.env.example` 为 `.env` 后填写以下两项。

- `ANTHROPIC_API_KEY`:必填,Anthropic 控制台申请的 API 密钥
- `PROXY`:可选,出站 HTTP / HTTPS 代理地址。留空则直连;填写后进程内所有出站 fetch 与常见 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量都会指向该代理。示例值 `http://127.0.0.1:15236`

## 启动

1. 复制 `.env.example` 为 `.env`,按上节填写变量
2. `npm install`
3. `npm run dev`

