// sis/static/arena/js/services/api_client.js
export class ApiClient {
  async fetchEvents(runId) {
    return await fetch(`/arena/api/events?run_id=${encodeURIComponent(runId)}`);
  }
  
  async fetchProvenance(caseId) {
    return await fetch(`/arena/api/provenance/${caseId}`);
  }
  
  async rescoreCase(caseId) {
    return await fetch(`/arena/api/scorables/${caseId}/rescore`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
  }
}