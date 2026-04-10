const std = @import("std");

pub const ncclResult_t = c_int;
pub const ncclSuccess: ncclResult_t = 0;
pub const ncclComm_t = *anyopaque;
pub const ncclDataType_t = c_int;
pub const ncclInt8: ncclDataType_t = 0;
pub const ncclChar: ncclDataType_t = 0;
pub const ncclUint8: ncclDataType_t = 1;
pub const ncclInt32: ncclDataType_t = 2;
pub const ncclInt: ncclDataType_t = 2;
pub const ncclUint32: ncclDataType_t = 3;
pub const ncclInt64: ncclDataType_t = 4;
pub const ncclUint64: ncclDataType_t = 5;
pub const ncclFloat16: ncclDataType_t = 6;
pub const ncclHalf: ncclDataType_t = 6;
pub const ncclFloat32: ncclDataType_t = 7;
pub const ncclFloat: ncclDataType_t = 7;
pub const ncclFloat64: ncclDataType_t = 8;
pub const ncclDouble: ncclDataType_t = 8;
pub const ncclBfloat16: ncclDataType_t = 9;
pub const ncclFp8E4M3: ncclDataType_t = 10;
pub const ncclFp8E5M2: ncclDataType_t = 11;
pub const ncclNumTypes: ncclDataType_t = 12;

pub const ncclRedOp_t = c_int;
pub const ncclSum: ncclRedOp_t = 0;
pub const ncclProd: ncclRedOp_t = 1;
pub const ncclMax: ncclRedOp_t = 2;
pub const ncclMin: ncclRedOp_t = 3;
pub const ncclAvg: ncclRedOp_t = 4;
pub const ncclNumOps: ncclRedOp_t = 5;

pub extern "c" fn ncclGetVersion(version: *c_int) ncclResult_t;
pub extern "c" fn ncclGetUniqueId(uniqueId: *ncclUniqueId) ncclResult_t;
pub extern "c" fn ncclCommInitRank(comm: *ncclComm_t, nranks: c_int, commId: ncclUniqueId, rank: c_int) ncclResult_t;
pub extern "c" fn ncclCommInitAll(comm: *ncclComm_t, ndev: c_int, devlist: [*c]c_int) ncclResult_t;
pub extern "c" fn ncclCommDestroy(comm: ncclComm_t) ncclResult_t;
pub extern "c" fn ncclCommAbort(comm: ncclComm_t) ncclResult_t;
pub extern "c" fn ncclCommGetAsyncError(comm: ncclComm_t, asyncError: *ncclResult_t) ncclResult_t;
pub extern "c" fn ncclCommCount(comm: ncclComm_t, count: *c_int) ncclResult_t;
pub extern "c" fn ncclCommCuDevice(comm: ncclComm_t, device: *c_int) ncclResult_t;
pub extern "c" fn ncclCommUserRank(comm: ncclComm_t, rank: *c_int) ncclResult_t;
pub extern "c" fn ncclAllReduce(sendbuff: *const anyopaque, recvbuff: *anyopaque, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: *anyopaque) ncclResult_t;
pub extern "c" fn ncclBroadcast(sendbuff: *const anyopaque, recvbuff: *anyopaque, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: *anyopaque) ncclResult_t;
pub extern "c" fn ncclReduce(sendbuff: *const anyopaque, recvbuff: *anyopaque, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, root: c_int, comm: ncclComm_t, stream: *anyopaque) ncclResult_t;
pub extern "c" fn ncclAllGather(sendbuff: *const anyopaque, recvbuff: *anyopaque, sendcount: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: *anyopaque) ncclResult_t;
pub extern "c" fn ncclReduceScatter(sendbuff: *const anyopaque, recvbuff: *anyopaque, recvcount: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: *anyopaque) ncclResult_t;
pub extern "c" fn ncclSend(sendbuff: *const anyopaque, count: usize, datatype: ncclDataType_t, peer: c_int, comm: ncclComm_t, stream: *anyopaque) ncclResult_t;
pub extern "c" fn ncclRecv(recvbuff: *anyopaque, count: usize, datatype: ncclDataType_t, peer: c_int, comm: ncclComm_t, stream: *anyopaque) ncclResult_t;
pub extern "c" fn ncclGroupStart() ncclResult_t;
pub extern "c" fn ncclGroupEnd() ncclResult_t;

pub const ncclUniqueId = extern struct {
    internal: [128]u8,
};

pub const NcclCommunicator = struct {
    comm: ncclComm_t,
    rank: c_int,
    nranks: c_int,
    
    pub fn init(rank: c_int, nranks: c_int, unique_id: ncclUniqueId) !NcclCommunicator {
        var comm: ncclComm_t = undefined;
        const result = ncclCommInitRank(&comm, nranks, unique_id, rank);
        if (result != ncclSuccess) {
            return error.NcclInitFailed;
        }
        return NcclCommunicator{
            .comm = comm,
            .rank = rank,
            .nranks = nranks,
        };
    }
    
    pub fn deinit(self: *NcclCommunicator) void {
        _ = ncclCommDestroy(self.comm);
    }
    
    pub fn allReduce(self: *NcclCommunicator, sendbuf: *const anyopaque, recvbuf: *anyopaque, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, stream: *anyopaque) !void {
        const result = ncclAllReduce(sendbuf, recvbuf, count, datatype, op, self.comm, stream);
        if (result != ncclSuccess) {
            return error.NcclAllReduceFailed;
        }
    }
    
    pub fn allGather(self: *NcclCommunicator, sendbuf: *const anyopaque, recvbuf: *anyopaque, sendcount: usize, datatype: ncclDataType_t, stream: *anyopaque) !void {
        const result = ncclAllGather(sendbuf, recvbuf, sendcount, datatype, self.comm, stream);
        if (result != ncclSuccess) {
            return error.NcclAllGatherFailed;
        }
    }
    
    pub fn reduceScatter(self: *NcclCommunicator, sendbuf: *const anyopaque, recvbuf: *anyopaque, recvcount: usize, datatype: ncclDataType_t, op: ncclRedOp_t, stream: *anyopaque) !void {
        const result = ncclReduceScatter(sendbuf, recvbuf, recvcount, datatype, op, self.comm, stream);
        if (result != ncclSuccess) {
            return error.NcclReduceScatterFailed;
        }
    }
    
    pub fn broadcast(self: *NcclCommunicator, sendbuf: *const anyopaque, recvbuf: *anyopaque, count: usize, datatype: ncclDataType_t, root: c_int, stream: *anyopaque) !void {
        const result = ncclBroadcast(sendbuf, recvbuf, count, datatype, root, self.comm, stream);
        if (result != ncclSuccess) {
            return error.NcclBroadcastFailed;
        }
    }
    
    pub fn send(self: *NcclCommunicator, sendbuf: *const anyopaque, count: usize, datatype: ncclDataType_t, peer: c_int, stream: *anyopaque) !void {
        const result = ncclSend(sendbuf, count, datatype, peer, self.comm, stream);
        if (result != ncclSuccess) {
            return error.NcclSendFailed;
        }
    }
    
    pub fn recv(self: *NcclCommunicator, recvbuf: *anyopaque, count: usize, datatype: ncclDataType_t, peer: c_int, stream: *anyopaque) !void {
        const result = ncclRecv(recvbuf, count, datatype, peer, self.comm, stream);
        if (result != ncclSuccess) {
            return error.NcclRecvFailed;
        }
    }
    
    pub fn groupStart() !void {
        const result = ncclGroupStart();
        if (result != ncclSuccess) {
            return error.NcclGroupStartFailed;
        }
    }
    
    pub fn groupEnd() !void {
        const result = ncclGroupEnd();
        if (result != ncclSuccess) {
            return error.NcclGroupEndFailed;
        }
    }
};

pub fn ncclInit() !void {
    var version: c_int = 0;
    const result = ncclGetVersion(&version);
    if (result != ncclSuccess) {
        return error.NcclInitFailed;
    }
}

pub fn ncclCleanup() void {
}

pub fn ncclGetUniqueId() !ncclUniqueId {
    var unique_id: ncclUniqueId = undefined;
    const result = ncclGetUniqueId(&unique_id);
    if (result != ncclSuccess) {
        return error.NcclGetUniqueIdFailed;
    }
    return unique_id;
}
