# From Impressions to Attention

*Measuring billboard audiences with computer vision · AI in Marketing*

**Marketing takeaway.** Use traffic counts to shortlist placements, then collect attention and pricing data before booking. This demo uses recorded traffic, a head-direction test and published examples of campaign performance.

## 1. Traffic and opportunities near the screen

| Measure | The Tenor South, Toronto | Piccadilly Lights, London |
|---|---:|---:|
| Conditions in the clip | Daytime, right to left | Nighttime, left to right |
| Common observation window | 11.91 s | 11.91 s |
| Vehicle crossings detected by YOLO11m and ByteTrack | 4 | 7 |
| Crossings matched to the marked potential viewing zone | 4 | 7 |
| Matched visits with observed entry and exit | 1 of 4 | 0 of 7 |
| User’s manual check and AI count | 1 each from 3.50 to 4.50 s | 1 each from 8.50 to 11.50 s |

The clips sample different road approaches, lighting and camera positions. The manual checks cover one clear crossing per clip. The full counts are AI estimates. One Toronto car spent 2.50 seconds in the marked zone from entry to exit. A comparison of typical viewing times would require more complete visits. The Bay Area highway clip served as an initial vehicle-counting test. [Toronto footage](https://www.pexels.com/video/dynamic-urban-intersection-with-billboards-34602127/) · [London footage](https://www.pexels.com/video/illuminated-billboard-along-the-road-2112773/) · [Highway footage](https://www.pexels.com/video/traffic-on-the-580-freeway-san-francisco-bay-area-19347491/)

**Expected audience vs. observed traffic.** Branded Cities cited average daily circulation of 158,540 people for the Yonge and Dundas area in its 2020 launch announcement. The Toronto clip covers one marked approach for 11.91 seconds and counts vehicles. The figures differ in unit, period and coverage. For a valid comparison, request the seller’s forecast for the exact display and time window, define whether it measures circulation or opportunities to see, and compare it with the same independently measured metric. [Branded Cities launch announcement](https://brandedcities.com/2020/01/101278/)

## 2. What the attention test established

In a close-up video, the AI matched the manual head-direction labels on **58 of 58 sampled frames** in five stable intervals from one subject. The labels describe whether the person faced the camera or turned away. In the Toronto street clip, the faces detected on 31 of 77 pedestrian tracks fell below the 40-pixel size threshold. This is why the demo uses the close-up for head-direction scoring. [Head-turn footage](https://www.pexels.com/video/a-man-turning-his-head-sideways-5474326/)

In the BIWI benchmark, mean absolute head-pose error was **4.41°** on 144 accepted frames. On the same frames, error was **4.64°** at 40 pixels and **6.54°** at 20 pixels. Applying head-direction estimation to a billboard requires visible faces and a measured direction from each person to the display. [BIWI dataset](https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose)

## 3. What a real campaign showed about paid visits

In a separate published campaign at Edmonton’s Muttart Conservatory, paid attendance per open day was **57.9% higher** than in the corresponding period a year earlier. The difference between the two venues’ year-over-year rates widened by **88.7 percentage points** from before to during the campaign. The gap was already widening before the ads began. These figures describe the recorded attendance and leave the campaign’s incremental effect unknown. Toronto and London have no sales records matched to the footage. [Bhargava and Donthu (1999), Table 1](https://www.researchgate.net/profile/Naveen-Donthu/publication/285666454_Sales_response_to_outdoor_advertising/links/64bf53ddb9ed6874a54269a3/Sales-Response-to-Outdoor-Advertising.pdf#page=6)

## 4. Marketing formulas

**CPM = ad spend ÷ impressions × 1,000.** In a separate published case, Blip reported 339 estimated impressions per dollar for DeeLorri Apparel’s one-billboard campaign. Inverting that rounded figure and multiplying by 1,000 gives **1,000 ÷ 339 ≈ $2.95 per 1,000 estimated impressions**. This uses the media owner’s audience estimate for that campaign. [Blip’s DeeLorri Apparel case](https://www.blipbillboards.com/success-stories/deelorri-apparel/)

**Attentive CPM = the same campaign spend ÷ AI-estimated attentive exposures × 1,000.** An attentive exposure here is a recorded occasion when someone appeared to face the screen while the ad played. For a live campaign, match the video timestamps to the screen’s playback log and use an attention estimate that covers the same booking as the spend. The head-direction test demonstrates the AI method on separate footage.

| Metric | Calculation | Data to collect |
|---|---|---|
| Scan rate | QR scans ÷ estimated impressions × 100 | Scans and estimated impressions for the same display and period |
| Purchase conversion rate | Tracked visits with an order ÷ tracked visits × 100 | QR landing-page visits and which visits resulted in orders |
| CPA (cost per order) | Ad spend ÷ tracked orders | Campaign spend and orders attributed to the campaign |
| ROAS | Attributed net revenue ÷ ad spend | Net revenue from those orders and campaign spend |

For a clickable ad, **CTR = clicks ÷ impressions × 100**. A display-specific QR code measures scan response for a billboard and can link visits to tracked orders. Estimating how many purchases the campaign added requires a comparison group. [Google Ads CTR definition](https://support.google.com/google-ads/answer/2615875?hl=en) · [Google Ads CPA definition](https://support.google.com/google-ads/answer/6396841?hl=en)
