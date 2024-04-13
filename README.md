# 實現具可置換髮型的大頭貼素描圖像雛型
- ## 簡介
  > 社交媒體是現代一種流行的趨勢。在這個網路時代，我們展示自己的方式之一就是透過在線上使用大頭貼圖像。在大頭貼生成工具中改變髮型部分，現有的工具只把要改變的髮型貼上，此技術限制了更換髮型的變換。因此本研究提出髮型可在頭部影像庫中挑選後再置換使用者的頭部影像的髮型。突破傳統僅貼上新髮型的限制。此外，本研究也應用Cycle-GAN 的技術，將人頭影像轉換成素描圖像，增進在網路媒體使用大頭貼的個人特性。
- ## 相關研究
  > 從人頭影像萃取頭髮後的空白區需換成額頭的膚色，才可解決新舊髮型長短不一的問題。98年周家凱的碩士論文 提出，使用臉頰皮膚來填充額頭的部分，使頭髮蓋上去時不會留下空白。由於臉頰膚色和鄰近頭髮的額頭相對而言比較亮，所以若直接用臉頰膚色填補原頭髮移除後的空白區會造成膚色和額頭部一致的缺點。因此本研究提出利用緊鄰頭髮的額頭膚色取代原頭髮移除後的空白區。
  > 
  > Cycle-GAN 由Jun-Yan Zhu 等人設計的神經網路結構，可以不破壞圖片結構將顏色進行轉換，例如馬轉成斑馬、照片轉成水彩畫等等。本研究應用此技術將人頭影像轉換成素描圖來增加個人特色。

- ## 開發環境與使用套件
  > 程式編譯：VS-Code
  >
  > 網頁開發：HTML CSS JavaScript Flask
  >
  > 影像處理：OpenCV K-means CV-zone D-lib Cycle-GAN
- ## 功能
  - 應用D-lib取得人臉特徵前處理 : 圖片正規化
  - 頭髮遮罩和無頭髮圖像：使用UV和初始群心
  - 頭髮製造區的膚色置換：膚色置換、柏松融合
  - 髮型提取置換、自動調整大小：節省時間
  - 大頭貼素描： 使用 Cycle-GAN 來實現
  - 使用者介面：讓用戶者輕鬆使用
- ## 內容
  > Models下載位置：
  >
  > 將 Models 解壓縮至 cyclegan 資料夾底下
- ### 圖像正規化
```
```
- ### 頭髮區域的遮罩
  - k-mean初始群心
  - 分水嶺分析
  - 頭髮群的擇定
  - 還原眉毛

- ### 頭髮提取並恢復劉海
```
```
- ### 模擬頭部
  - 建立目標參考點
  - 置換頭髮的前處理
  - 頭髮膚色轉換

- ### 自動調整髮型位置大小
```
```
- ### 應用Cycle-GAN使大頭貼手繪素描
  > 參考論文(Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks) ：https://junyanz.github.io/CycleGAN/
  > 
  > 論文所提出的神經網路，此神經網路擅長將圖像外顏色進行轉換例如馬轉斑馬等，且轉換圖片不需要成對也可以訓練

- ### 介面設計
```
```
- ## 參考文獻
  [1]	Google Play Store Hair try-on - hair styling

  [2]	CycleGAN由Jun-Yan Zhu、Taesung Park、Phillip Isola 、Alexei A. Efros

  [3]	周家凱的碩士論文-Virtual Haircut and Hairstyle Cloning

  [4]	香港中文大學人臉草圖資料庫（CUFS）

  [5]	MaskGAN: Towards Diverse and Interactive Facial Image Manipulation由Cheng-Han Lee、 Ziwei Liu、 Lingyun Wu1 Ping Luo
