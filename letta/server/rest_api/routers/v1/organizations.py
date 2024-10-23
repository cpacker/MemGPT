from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from letta.schemas.organization import Organization, OrganizationCreate
from letta.server.rest_api.utils import get_letta_server

if TYPE_CHECKING:
    from letta.server.server import SyncServer


router = APIRouter(prefix="/orgs", tags=["organization", "admin"])


@router.get("/", tags=["admin"], response_model=List[Organization], operation_id="list_orgs")
def get_all_orgs(
    cursor: Optional[str] = Query(None),
    limit: Optional[int] = Query(50),
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Get a list of all orgs in the database
    """
    try:
        next_cursor, orgs = server.organization_manager.list_organizations(cursor=cursor, limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return orgs


@router.post("/", tags=["admin"], response_model=Organization, operation_id="create_organization")
def create_org(
    request: OrganizationCreate = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Create a new org in the database
    """
    org = server.organization_manager.create_organization(request)
    return org


@router.delete("/", tags=["admin"], response_model=Organization, operation_id="delete_organization_by_id")
def delete_org(
    org_id: str = Query(..., description="The org_id key to be deleted."),
    server: "SyncServer" = Depends(get_letta_server),
):
    # TODO make a soft deletion, instead of a hard deletion
    try:
        org = server.organization_manager.get_organization_by_id(org_id=org_id)
        if org is None:
            raise HTTPException(status_code=404, detail=f"Organization does not exist")
        server.organization_manager.delete_organization_by_id(org_id=org_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return org
