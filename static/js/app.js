let currentId = null;

function runSearch() {
  const query = $("#query").val();
  const mode = $("#mode").val();

  if (!query) return;

  $.ajax({
    url: "/api/search",
    method: "POST",
    contentType: "application/json",
    data: JSON.stringify({ query, mode }),
    success: function(results) {
      $("#results").empty();
      results.forEach(item => {
        const li = $("<li>").text(`${item.title} (${item.timestamp})`).data("item", item);
        $("#results").append(li);
      });
    }
  });
}

function loadMemory(item) {
  currentId = item.id;
  $("#memory-title").text(item.title);
  $("#preview").html(`<strong>USER:</strong><br>${item.user_text}<br><br><strong>AI:</strong><br>${item.ai_text}`);
  $("#tags").val((item.tags || []).join(", "));
  $("#summary").val(item.summary || "");
}

function saveMemory() {
  if (!currentId) return;

  const tags = $("#tags").val().split(",").map(t => t.trim()).filter(Boolean);
  const summary = $("#summary").val();

  $.ajax({
    url: `/api/memory/${currentId}`,
    method: "PATCH",
    contentType: "application/json",
    data: JSON.stringify({ tags, summary }),
    success: () => alert("✅ Saved!")
  });
}

function exportMemory() {
  if (!currentId) return;

  $.get(`/api/export/${currentId}`, function(data) {
    const blob = new Blob([data.markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${data.json.title.replace(/ /g, "_")}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });
}

$(document).ready(function () {
  $("#query").on("keydown", function (e) {
    if (e.key === "Enter") runSearch();
  });

  $("#results").on("click", "li", function () {
    const item = $(this).data("item");
    loadMemory(item);
  });

  $("#save").click(saveMemory);
  $("#export").click(exportMemory);
});
