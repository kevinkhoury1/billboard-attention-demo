import json
from pathlib import Path

import streamlit as st


OUTPUTS = Path(__file__).resolve().parents[1] / "data" / "outputs"
SITES = {
    "toronto_01": ("Toronto", "The Tenor South", "Day · Right to left", "phase2_toronto/count_review.json"),
    "london_01": ("London", "Piccadilly Lights", "Night · Left to right", "phase4/london_count_check.json"),
}
STUDY_TABLE = (
    "https://www.researchgate.net/profile/Naveen-Donthu/publication/285666454_Sales_response_to_outdoor_advertising/"
    "links/64bf53ddb9ed6874a54269a3/Sales-Response-to-Outdoor-Advertising.pdf#page=6"
)
DEELORRI_CASE = "https://www.blipbillboards.com/success-stories/deelorri-apparel/"


def read_result(path):
    return json.loads((OUTPUTS / path).read_text(encoding="utf-8"))


def show_sites(comparison):
    duration = comparison["window_s"]["end_exclusive"] - comparison["window_s"]["start"]
    st.subheader("How much traffic passed each screen?")
    st.caption(f"Same {duration:.2f}-second window per clip · One marked direction at each site")
    sites = {SITES[site["site_id"]][0]: site for site in comparison["sites"]}
    for column, (city, site) in zip(st.columns(len(sites)), sites.items()):
        _, display, context, _ = SITES[site["site_id"]]
        with column.container(border=True):
            st.markdown(f"#### {city} · {display}")
            st.caption(context)
            st.metric("AI-counted vehicle crossings", site["vehicle_crossings"])
            complete = site["complete_matched_zone_visits"]
            matched = site["crossings_matched_to_zone"]
            dwell = site["median_complete_matched_zone_time_s"]
            st.write(f"**{matched}** crossings matched the potential viewing zone.")
            st.write(f"**{complete} of {matched}** matched visits tracked from entry to exit.")
            if dwell is not None:
                st.write(f"Median time in zone for complete visits · **{dwell:.2f} s**")
            st.markdown(f"[Watch the source clip]({site['source']['source_url']})")

    st.info("London had more crossings in its clip. Road coverage and lighting differ, and only one Toronto car had "
            "a complete visit. These samples give us too little evidence to rank the placements.")

    with st.expander("Check the manual counts and crossing timestamps"):
        city = st.selectbox("Review a placement", list(sites), key="review_city")
        selected = sites[city]
        review = read_result(SITES[selected["site_id"]][3])["windows"][0]
        st.markdown(f"**{review['start_s']:.2f}–{review['end_s']:.2f} s** · "
                    f"{review['manual_confirmed']} manual and {review['ai_crossings']} AI")
        st.caption(review["notes"])
        st.dataframe([
            {"Crossing time (s)": round(row["crossing_s"], 2), "Vehicle class": row["vehicle_class"].title(),
             "Matched zone visit": ("Complete" if row["complete_within_window"] else
                                    "Partial" if row["zone_visit"] else "None")}
            for row in selected["crossing_details"]
        ], hide_index=True, width="stretch")
        st.caption("The timestamp marks when the bottom centre of the tracked vehicle box crossed the line. "
                   "The manual check covers only the interval above.")


def show_attention(attention, biwi, head_turn):
    st.subheader("Can the AI tell which way someone is facing?")
    with st.container(border=True):
        st.caption("Close-up head-direction test · One subject")
        st.metric("Sampled frames that matched manual labels",
                  f"{head_turn['agreeing_assessable_frames']} of {head_turn['assessable_frames']}")
        st.write(f"The AI distinguished facing the camera from turning away across "
                 f"{len(head_turn['windows'])} manually labelled intervals where the head was held steady.")
        st.markdown(f"[Watch the head-turn clip]({head_turn['source']})")

    st.markdown("#### Benchmark on the BIWI public dataset")
    native = biwi["native_resolution"]
    stress = biwi["resolution_stress_test"]["reduced_face_width_px"]
    st.caption(f"6DRepNet mean absolute head-pose error, averaged over pitch, yaw and roll. "
               f"The same {native['frames']} accepted faces were used for each resolution.")
    columns = st.columns(3, border=True)
    for column, label, sample in zip(columns, ("Native resolution", "40 px face width", "20 px face width"),
                                     (native, stress["40"], stress["20"])):
        column.metric(label, f"{sample['mean_absolute_error_deg']:.2f}°")
    st.markdown(f"[BIWI dataset]({biwi['dataset_terms']})")
    st.caption(f"The Toronto street clip yielded {attention['pedestrian_tracks']} pedestrian tracks. The AI "
               f"detected a face on {attention['pedestrian_tracks_with_a_detected_face']}. The faces were "
               f"below the {attention['settings']['min_face_width_px']} px quality threshold, so the demo uses "
               "the close-up clip for head-direction scoring.")


def show_campaign(observations, analysis):
    st.subheader("Did a real billboard campaign generate paid visits?")
    st.caption("Edmonton · Muttart Conservatory · Separate published campaign")
    reported, contrast = st.columns(2, border=True)
    reported.metric("Muttart paid attendance vs. prior year", f"{observations['periods']['during']['treated']:+.1f}%")
    contrast.metric("Change in the gap between venues", f"{analysis['descriptive_difference_in_differences_pp']:+.1f} pp")
    chart = OUTPUTS / "phase5/muttart_paid_attendance.png"
    st.image(str(chart), width="stretch", caption="Source data · Bhargava and Donthu (1999), Table 1")
    gaps = analysis["weekly_pre_campaign_gap_pp"]
    st.info(f"The gap grew from **{gaps['1']:.1f} to {gaps['4']:.1f} percentage points** before the ads began. "
            "The exact increase caused by the campaign is unknown. The Toronto and London clips have no matching sales records.")
    st.download_button("Download attendance chart", chart.read_bytes(), chart.name, "image/png")
    st.markdown(f"[Original study table]({STUDY_TABLE})")


def show_marketing_formulas():
    st.subheader("How much did the billboard cost per 1,000 impressions?")
    impressions_per_dollar = 339  # Rounded figure published by Blip for the DeeLorri campaign.
    cpm = 1_000 / impressions_per_dollar
    with st.container(border=True):
        st.caption("DeeLorri Apparel · One digital billboard · Separate published case")
        st.metric("Approximate cost per 1,000 estimated impressions", f"${cpm:.2f}")
        st.markdown("**CPM = ad spend ÷ impressions × 1,000**")
        st.write(f"Blip reported {impressions_per_dollar} estimated impressions per dollar. Divide 1,000 by "
                 f"{impressions_per_dollar} to get about ${cpm:.2f} per 1,000 estimated impressions.")
        st.markdown(f"[Blip’s published campaign data]({DEELORRI_CASE})")

    with st.container(border=True):
        st.markdown("#### Standard CPM vs. Attentive CPM")
        standard, attentive = st.columns(2)
        with standard:
            st.markdown("**Standard CPM**")
            st.write("Campaign spend ÷ estimated impressions × 1,000")
            st.caption("What did we pay per 1,000 estimated opportunities to see the ad?")
        with attentive:
            st.markdown("**Attentive CPM with our AI**")
            st.write("Same campaign spend ÷ AI-estimated attentive exposures × 1,000")
            st.caption("What would we pay per 1,000 times someone appeared to face the screen while our ad played?")
        st.write("For a live campaign, match the video timestamps to the screen’s ad playback log. "
                 "Use the attention estimate and spend for that same booking.")
        st.caption("The $2.95 result above comes from DeeLorri’s published campaign. "
                   "The Attention test demonstrates head direction on separate footage.")

    st.markdown("#### Measuring response to a billboard with a QR code")
    st.markdown("""
| Metric | Calculation | Data to collect |
|---|---|---|
| Scan rate | QR scans ÷ estimated impressions × 100 | Scans and estimated impressions for the same display and period |
| Purchase conversion rate | Tracked visits with an order ÷ tracked visits × 100 | Visits to the QR landing page and which visits resulted in orders |
| CPA (cost per order) | Ad spend ÷ tracked orders | Campaign spend and orders attributed to that campaign |
| ROAS | Attributed net revenue ÷ ad spend | Net revenue from those orders and campaign spend |
""")
    st.caption("For a clickable ad, CTR uses clicks ÷ impressions × 100. Scan rate measures the QR response to a billboard.")
    st.info("Use traffic counts to shortlist placements, then collect attention and pricing data before booking.")
    report = OUTPUTS / "phase6/final_report.md"
    st.download_button("Download final marketing report", report.read_bytes(), report.name, "text/markdown", type="primary")


def main():
    st.set_page_config(page_title="From Impressions to Attention", layout="wide")
    st.markdown("""
        <style>
            .block-container {max-width: 72rem; padding-top: 2.6rem;}
            .project-hero {background: #133b3a; color: #f2fcfa; padding: clamp(1.2rem, 4vw, 2.6rem);
                           border-radius: 1rem; margin-bottom: 1.1rem;}
            .project-hero h1 {color: inherit; font-size: clamp(1.8rem, 5vw, 3rem); line-height: 1.12;
                              margin: 0.5rem 0; padding: 0;}
            .project-hero p {max-width: 46rem; margin: 0; color: #d0e7e3;}
            .project-hero .project-kicker {color: #a5e6d9; font-size: 0.75rem; font-weight: 700;
                                           letter-spacing: 0.08em;}
        </style>
        <div class="project-hero">
            <div class="project-kicker">AI IN MARKETING · FIELD STUDY</div>
            <h1>From Impressions to Attention</h1>
            <p>What traffic footage can tell an advertiser, and which decisions need campaign data.</p>
        </div>
    """, unsafe_allow_html=True)
    placement, attention, campaign, formulas = st.tabs([
        "Site comparison", "Attention test", "Campaign impact", "Marketing formulas"
    ])
    try:
        with placement:
            show_sites(read_result("phase4/comparison.json"))
        with attention:
            show_attention(read_result("phase3_toronto/attention.json"), read_result("phase3_toronto/biwi_pose.json"),
                           read_result("phase3_toronto/online_head_turn_check.json"))
        with campaign:
            show_campaign(read_result("phase5/muttart_observations.json"), read_result("phase5/muttart_analysis.json"))
        with formulas:
            show_marketing_formulas()
    except FileNotFoundError as error:
        missing = Path(error.filename).relative_to(OUTPUTS)
        st.error(f"Evidence file missing · data/outputs/{missing.as_posix()}. Restore the saved project result and reload.")


if __name__ == "__main__":
    main()
