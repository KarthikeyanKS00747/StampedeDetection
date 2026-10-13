import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "outputs";
await fs.mkdir(outputDir, { recursive: true });
const workbook = Workbook.create();
const summary = workbook.worksheets.add("Review summary");
const candidates = workbook.worksheets.add("Candidates");

const headers = ["ID", "Candidate", "Purpose", "Event / motion", "Source type", "Source URL", "Rights status", "Graphic risk", "Download status", "Reviewer decision", "Reviewer notes", "View / camera", "Crowd condition", "Suggested tier", "AI recommendation"];
const rows = [
  ["UMN-01", "UMN Crowd Activity", "Baseline motion anomaly", "Normal walking then collective escape", "Official academic benchmark", "https://www.crcv.ucf.edu/projects/Abnormal_Crowd/", "Research benchmark; retain attribution", "Low", "Downloaded", "Approve", "Use for High Risk motion validation only; not a real stampede label", "Fixed overhead / mixed", "Moderate density then collective escape", "High Risk only", "Keep"],
  ["EVD-01", "Evacuation Videos Database", "Find non-graphic evacuation examples", "Emergency evacuation and crowd behaviour", "Research dataset", "https://figshare.com/articles/dataset/Evacuation_Videos_Database/6974321", "Figshare record is CC BY 4.0; preserve attribution and do not redistribute source footage", "Variable", "Not downloaded", "Approve", "Select only reviewed pre-incident/onset clips", "Mixed", "Mixed; review clip-by-clip", "Normal to High Risk", "Keep after review"],
  ["DRILL-01", "GPD Riot Drill 2009", "Flow/egress control baseline", "Organised crowd-control drill", "Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:GPD_RIOT_DRILL_2009.webm", "CC BY 3.0; attribute SOFISINTOWN", "Low", "Not downloaded", "Approve", "Useful only as controlled-drill data, not crowd-crush evidence", "Likely ground level", "Organised drill", "Normal / Moderate", "Keep"],
  ["YTN-01", "Itaewon news/CCTV report", "Case-study reference only", "Reported pushing and crowd crush reconstruction", "Verified news YouTube channel", "https://www.youtube.com/watch?v=CmjnJX5dwaM", "Copyrighted news material; no download/use without permission", "High", "Not downloaded", "Reference only", "Do not use if graphic or if rights cannot be cleared", "News/CCTV report", "Severe crowd crush", "No training label", "Do not download"],
  ["AP-01", "Itaewon CCTV / forensic reporting", "Context and density reference", "Crowd-crush investigation; reporting says 180+ videos were analysed", "Associated Press report", "https://apnews.com/article/33331e09fecf4dd321d72c9d5dc92eff", "Article/reference only; footage not licensed", "Medium", "Not downloadable", "Reference only", "Use only for project background and calibration discussion", "N/A", "Reported crowd density evidence", "No training label", "Keep as citation only"],
  ["AP-02", "Maha Kumbh incident reporting", "Recent case-study reference", "Large-gathering crowd loss of control", "Associated Press report", "https://www.ap.org/news-highlights/spotlights/2025/police-say-at-least-30-people-have-died-in-a-stampede-at-the-massive-maha-kumbh-festival-in-india/", "Article/reference only; footage not licensed", "High", "Not downloadable", "Reference only", "Do not source graphic footage from this event", "N/A", "Reported crowd loss of control", "No training label", "Keep as citation only"],
  ["PEX-01", "Aerial outdoor concert", "Dense crowd and geometry baseline", "Large concert audience", "Pexels stock video", "https://www.pexels.com/video/a-large-crowd-of-people-gathered-in-front-of-a-stage-17758188/", "Pexels free licence; retain source record", "Low", "Not downloaded", "Approve", "Strong overhead view and 41-second duration", "Aerial / overhead", "Dense but orderly", "Moderate", "High value"],
  ["PEX-02", "Outdoor concert crowd", "Short normal/moderate baseline", "Concert audience", "Pexels stock video", "https://www.pexels.com/video/crowd-on-concert-in-city-13658613/", "Pexels free licence; retain source record", "Low", "Not downloaded", "Approve", "Only 9 seconds, so use as small augmentation", "Ground / wide", "Orderly concert crowd", "Normal / Moderate", "Useful but short"],
  ["FOOT-01", "Football supporters", "Sports-crowd baseline", "Crowd of football supporters", "Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:Football_supporters.webm", "CC BY 2.0 stated; page notes licence review is still needed", "Low", "Not downloaded", "Pending", "Technically useful, but manually verify licensing note before keeping", "Ground / crowd-facing", "Supporters in stadium context", "Normal / Moderate", "Review licence note"],
  ["APM-01", "Argentina fans greet Messi", "Extreme-scale event reference", "Hundreds of thousands gather for World Cup celebration", "Associated Press YouTube", "https://www.youtube.com/watch?v=QF5s5-ZS-JQ", "Copyrighted news footage; no download/use without permission", "Low", "Not downloaded", "Reference only", "Excellent scenario relevance, but not reusable training material", "Live wide broadcast", "Very large celebratory crowd", "No training label", "Use as case-study inspiration only"],
];

summary.showGridLines = false;
summary.getRange("A2:F2").merge();
summary.getRange("A2").values = [["Crowd Video Review"]];
summary.getRange("A3:F3").merge();
summary.getRange("A3").values = [["Karthikeyan review gate for candidate footage. Do not download copyrighted or graphic material."]];
summary.getRange("A5:B9").values = [
  ["Candidate count", rows.length],
  ["Downloaded", null],
  ["Approved", null],
  ["Rejected", null],
  ["Pending review", null],
];
summary.getRange("A11:F15").values = [
  ["Review rules", "", "", "", "", ""],
  ["1", "Approve only clear, non-graphic footage with usable camera angle and provenance.", "", "", "", ""],
  ["2", "Do not download YouTube/news footage unless the rights holder grants permission.", "", "", "", ""],
  ["3", "UMN is a motion-anomaly benchmark, not stampede ground truth.", "", "", "", ""],
  ["4", "Use final approved footage for reviewed risk windows, not automatic incident labels.", "", "", "", ""],
];
summary.getRange("B6:B9").formulas = [
  ["=COUNTIF(Candidates!$I$2:$I$100,\"Downloaded\")"],
  ["=COUNTIF(Candidates!$J$2:$J$100,\"Approve\")"],
  ["=COUNTIF(Candidates!$J$2:$J$100,\"Reject\")"],
  ["=COUNTIF(Candidates!$J$2:$J$100,\"Pending\")"],
];
summary.getRange("A2:F2").format = { font: { name: "Arial", size: 16, bold: true, color: "#17324D" } };
summary.getRange("A3:F3").format = { font: { name: "Arial", size: 10, italic: true, color: "#526B84" } };
summary.getRange("A5:B9").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
summary.getRange("A5:A9").format = { fill: "#DCE6F1", font: { name: "Arial", bold: true, color: "#17324D" } };
summary.getRange("B5:B9").format = { font: { name: "Arial", size: 11, bold: true }, horizontalAlignment: "center" };
summary.getRange("A11:F11").format = { fill: "#17324D", font: { name: "Arial", bold: true, color: "#FFFFFF" } };
summary.getRange("A11:F15").format.wrapText = true;
summary.getRange("A1:F15").format.font = { name: "Arial", size: 10 };
summary.getRange("A:A").format.columnWidth = 22;
summary.getRange("B:B").format.columnWidth = 70;
summary.getRange("C:F").format.columnWidth = 16;
summary.getRange("12:15").format.rowHeight = 28;

candidates.showGridLines = false;
candidates.getRangeByIndexes(0, 0, 1, headers.length).values = [headers];
candidates.getRangeByIndexes(1, 0, rows.length, headers.length).values = rows;
candidates.getRange(`A1:O${rows.length + 1}`).format.font = { name: "Arial", size: 10 };
candidates.getRange("A1:O1").format = { fill: "#17324D", font: { name: "Arial", bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
candidates.getRange(`A2:O${rows.length + 1}`).format = { verticalAlignment: "top", wrapText: true };
candidates.getRange(`A1:O${rows.length + 1}`).format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
candidates.getRange(`J2:J100`).dataValidation = { rule: { type: "list", values: ["Pending", "Approve", "Reject", "Reference only"] } };
candidates.getRange(`J2:J${rows.length + 1}`).conditionalFormats.add("containsText", { text: "Approve", format: { fill: "#D9EAD3", font: { color: "#274E13", bold: true } } });
candidates.getRange(`J2:J${rows.length + 1}`).conditionalFormats.add("containsText", { text: "Reject", format: { fill: "#F4CCCC", font: { color: "#990000", bold: true } } });
candidates.getRange(`J2:J${rows.length + 1}`).conditionalFormats.add("containsText", { text: "Pending", format: { fill: "#FFF2CC", font: { color: "#7F6000", bold: true } } });
candidates.freezePanes.freezeRows(1);
const widths = [12, 28, 25, 34, 23, 55, 36, 14, 19, 19, 42, 23, 27, 18, 22];
for (let col = 0; col < widths.length; col += 1) candidates.getRangeByIndexes(0, col, rows.length + 1, 1).format.columnWidth = widths[col];
candidates.getRange(`1:${rows.length + 1}`).format.rowHeight = 44;
candidates.getRange("A1:K1").format.rowHeight = 34;
candidates.tables.add(`A1:O${rows.length + 1}`, true, "CandidateVideosTable");

workbook.recalculate();
const inspect = await workbook.inspect({ kind: "table", range: "Candidates!A1:O11", include: "values,formulas", tableMaxRows: 12, tableMaxCols: 15 });
console.log(inspect.ndjson);
const preview = await workbook.render({ sheetName: "Candidates", range: "A1:O11", scale: 1, format: "png" });
await fs.writeFile(`${outputDir}/crowd_video_review_preview.png`, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}/crowd_video_review.xlsx`);
