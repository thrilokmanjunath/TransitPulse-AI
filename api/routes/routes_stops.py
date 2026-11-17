"""Route and Stop Catalog Endpoints."""

from fastapi import APIRouter, HTTPException, Query

from api.schemas import RouteResponse, StopResponse
from transitpulse.ingestion.transit_connector import TransitConnector

router = APIRouter(prefix="/api/v1", tags=["Transit Catalog"])
transit_conn = TransitConnector()


@router.get("/routes", response_model=list[RouteResponse])
def get_routes() -> list[RouteResponse]:
    """Retrieves all active BMTC corridor routes with distances and terminals."""
    df_routes = transit_conn.get_routes()
    routes = []
    for _, r in df_routes.iterrows():
        routes.append(
            RouteResponse(
                route_id=str(r["route_id"]),
                route_short_name=str(r["route_short_name"]),
                origin_name=str(r["origin_name"]),
                destination_name=str(r["destination_name"]),
                distance_km=float(r["distance_km"]),
                scheduled_duration_min=float(r["scheduled_duration_min"]),
                num_stops=int(r["num_stops"]),
            )
        )
    return routes


@router.get("/stops", response_model=list[StopResponse])
def get_stops(
    route_id: str = Query(..., description="BMTC Route ID (e.g., '335-E', '500-D')"),
) -> list[StopResponse]:
    """Retrieves the ordered sequence of bus stops and coordinates for a given route."""
    df_stops = transit_conn.get_stops()
    df_filtered = df_stops[df_stops["route_id"] == route_id].sort_values("stop_sequence")
    if df_filtered.empty:
        raise HTTPException(
            status_code=404, detail=f"Route '{route_id}' not found in BMTC catalog."
        )

    stops = []
    for _, s in df_filtered.iterrows():
        stops.append(
            StopResponse(
                route_id=str(s["route_id"]),
                stop_id=str(s["stop_id"]),
                stop_sequence=int(s["stop_sequence"]),
                stop_name=str(s["stop_name"]),
                stop_lat=float(s["stop_lat"]),
                stop_lon=float(s["stop_lon"]),
                scheduled_travel_time_sec=float(s["scheduled_travel_time_sec"]),
            )
        )
    return stops
