import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/mac/computerscience/23选题探索/bib/AI文本文献/AI文本文献综述v2.xlsx";
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const sheetNames = workbook.worksheets.items.map((sheet) => sheet.name);
console.log(JSON.stringify({ sheetNames }, null, 2));

for (const sheetName of sheetNames) {
  const inspect = await workbook.inspect({
    kind: "table",
    range: `${sheetName}!A1:Z12`,
    include: "values",
    tableMaxRows: 12,
    tableMaxCols: 26,
  });
  console.log(`\n### ${sheetName}`);
  console.log(inspect.ndjson);
}
